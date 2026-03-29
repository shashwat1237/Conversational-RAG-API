import os
import tempfile
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel


from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.schema import HumanMessage, AIMessage


GROQ_API_KEY = #hidden
SQLALCHEMY_DATABASE_URL = "sqlite:///./textify_production.db"


engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class DocumentMetadata(Base):
    __tablename__ = "uploaded_documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    upload_time = Column(DateTime, default=datetime.utcnow)
    file_size_bytes = Column(Integer)
    status = Column(String, default="processed")


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



app = FastAPI(title="Conversational RAG")

ACTIVE_VECTOR_DB = None


class ChatRequest(BaseModel):
    session_id: str
    query: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.post("/upload")
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    global ACTIVE_VECTOR_DB

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    try:
        file_bytes = file.file.read()

        db_doc = DocumentMetadata(
            filename=file.filename,
            file_size_bytes=len(file_bytes)
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)

       
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            path = tmp.name

        loader = PyPDFLoader(path)
        documents = loader.load()
        os.unlink(path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if ACTIVE_VECTOR_DB is None:
            ACTIVE_VECTOR_DB = FAISS.from_documents(chunks, embeddings)
        else:
            new_db = FAISS.from_documents(chunks, embeddings)
            ACTIVE_VECTOR_DB.merge_from(new_db)

        return {
            "status": "success",
            "document_id": db_doc.id
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat_with_memory(request: ChatRequest):
    global ACTIVE_VECTOR_DB

    if ACTIVE_VECTOR_DB is None:
        raise HTTPException(status_code=400, detail="Upload document first")

    try:
        
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )

        
        message_history = SQLChatMessageHistory(
            session_id=request.session_id,
            connection_string=SQLALCHEMY_DATABASE_URL,
            table_name="message_store"
        )

        
        retriever = ACTIVE_VECTOR_DB.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(request.query)

        context = "\n\n".join([doc.page_content for doc in docs])

        
        history_messages = message_history.messages

        history_text = ""
        for msg in history_messages:
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Assistant: {msg.content}\n"

        
        prompt = f"""
You are a helpful AI assistant.

Conversation so far:
{history_text}

Context from documents:
{context}

User question:
{request.query}

Instructions:
- Answer conversationally
- Use previous chat history if relevant
- If user asks about previous messages, answer correctly
"""

  
        response = llm.invoke(prompt)

        
        message_history.add_user_message(request.query)
        message_history.add_ai_message(response.content)

        return ChatResponse(
            answer=response.content,
            session_id=request.session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
