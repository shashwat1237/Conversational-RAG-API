# Conversational RAG API

A FastAPI-based backend that lets you upload PDF documents and have multi-turn conversations about their content. It combines Retrieval-Augmented Generation (RAG) with persistent chat memory, meaning the AI remembers what was said earlier in a session and uses that context alongside relevant document chunks to answer follow-up questions accurately.

---

## What It Does

1. You POST a PDF to `/upload` — the file is parsed, chunked, embedded, and merged into an in-memory FAISS vector index. Metadata (filename, size, upload time) is saved to SQLite.
2. You POST a question to `/chat` with a `session_id` — the app retrieves the top 4 most relevant document chunks, loads the full conversation history for that session from SQLite, builds a prompt combining both, and sends it to LLaMA 3.3 70B via Groq.
3. The user message and AI response are saved back to SQLite so the next turn has full context.

Multiple PDFs can be uploaded — their embeddings are merged into the same FAISS index so all documents are queryable together.

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| API Framework | FastAPI |
| PDF Parsing | LangChain `PyPDFLoader` |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (in-memory) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Chat Memory | LangChain `SQLChatMessageHistory` |
| Database | SQLite via SQLAlchemy |
| Data Validation | Pydantic |

---

## Project Structure

```
.
├── main.py                    # FastAPI application (all logic lives here)
├── textify_production.db      # SQLite DB (auto-created on first run)
└── README.md
```

---

## Database Schema

Two tables are auto-created on startup:

`uploaded_documents` — tracks every uploaded PDF:

| Column | Type | Description |
|---|---|---|
| id | Integer (PK) | Auto-incremented document ID |
| filename | String | Original filename |
| upload_time | DateTime | UTC timestamp of upload |
| file_size_bytes | Integer | Raw file size in bytes |
| status | String | Always `"processed"` |

`message_store` — managed by LangChain's `SQLChatMessageHistory`, stores all chat turns keyed by `session_id`.

---

## API Endpoints

### `POST /upload`

Uploads a PDF, processes it, and adds its embeddings to the active vector index.

Request: `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| file | File | Yes | A `.pdf` file |

Response:
```json
{
  "status": "success",
  "document_id": 1
}
```

Errors:
- `400` — file is not a PDF
- `500` — processing or DB error

---

### `POST /chat`

Sends a question and gets a context-aware, memory-backed answer.

Request body (JSON):
```json
{
  "session_id": "user-abc-123",
  "query": "What are the key findings in the report?"
}
```

| Field | Type | Description |
|---|---|---|
| session_id | string | Unique identifier for the conversation. Use the same ID across turns to maintain memory. |
| query | string | The user's question |

Response:
```json
{
  "answer": "The report highlights three key findings...",
  "session_id": "user-abc-123"
}
```

Errors:
- `400` — no document has been uploaded yet
- `500` — LLM or retrieval error

---

## Prerequisites

- Python 3.9+
- A valid [Groq API key](https://console.groq.com/)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/conversational-rag-api.git
cd conversational-rag-api

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
fastapi
uvicorn
sqlalchemy
pydantic
langchain
langchain-community
langchain-groq
faiss-cpu
sentence-transformers
pypdf
python-multipart
```

---

## Configuration

The Groq API key is hardcoded in the script. Before deploying or sharing, move it to an environment variable:

```python
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

Then set it before running:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

---

## Running the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

Interactive docs (Swagger UI) are auto-generated at `http://localhost:8000/docs`.

---

## Example Usage

### 1. Upload a PDF

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@report.pdf"
```

### 2. Ask a question

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session-001", "query": "Summarize the main points"}'
```

### 3. Ask a follow-up (same session_id preserves memory)

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session-001", "query": "What did you just say about the main points?"}'
```

---

## Key Implementation Details

### PDF Processing Pipeline

Each uploaded file is written to a `tempfile`, loaded with `PyPDFLoader`, then immediately deleted from disk. Text is split into 1000-character chunks with 200-character overlap to avoid losing context at chunk boundaries.

### Vector Index Merging

`ACTIVE_VECTOR_DB` is a module-level global. When the first PDF is uploaded it initializes the FAISS index. Subsequent uploads create a separate FAISS index and call `merge_from()` to fold it into the existing one, so all documents remain queryable from a single retriever.

### Persistent Conversation Memory

`SQLChatMessageHistory` stores every message turn in the `message_store` SQLite table, keyed by `session_id`. On each `/chat` call, the full history for that session is loaded, formatted as a plain-text transcript, and injected into the prompt before the document context and the new question.

### Prompt Construction

The prompt sent to the LLM is structured as:
1. Conversation history (all prior turns for the session)
2. Retrieved document chunks (top 4 by semantic similarity)
3. The current user question
4. Instructions to answer conversationally and reference history when relevant

### LLM Settings

`temperature=0.3` keeps responses factual and grounded rather than creative or hallucinated.

---

## Limitations

- The FAISS vector index is in-memory only — it is lost when the server restarts. Re-uploading documents is required after a restart.
- Only text-layer PDFs are supported. Scanned/image-based PDFs will produce empty or garbled output.
- The hardcoded API key must be replaced with an environment variable before any deployment.
- `ACTIVE_VECTOR_DB` is a single global — this design is not thread-safe under high concurrency. For production, use a persistent vector store (e.g., Chroma, Pinecone, pgvector).
- No authentication or rate limiting is implemented on the endpoints.

---

## License

MIT
