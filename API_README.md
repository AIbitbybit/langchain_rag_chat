# RAG API Service

A FastAPI service that provides an API for Retrieval-Augmented Generation (RAG) using LangChain.

## Setup

### Prerequisites

- Python 3.11+
- Poetry

### Installation

1. Install dependencies:

```bash
poetry install
```

2. Set up your environment variables:

```
# Create a copy of the existing .env file in rag_demos or create a new one
cp rag_demos/.env .env
```

3. Make sure your `.env` file contains at least:

```
OPENAI_API_KEY=your_openai_api_key
```

## Running the API

Start the API server:

```bash
poetry run python run_api.py
```

The API will be available at http://localhost:8000

## Stopping the API

There are multiple ways to stop the API server:

1. If you started the server in the foreground (terminal session is active):

   - Press `Ctrl+C` in the terminal where the server is running

2. If the server is running in the background:

   - Find the process ID:
     ```bash
     lsof -i :8000
     ```
   - Stop the process with the kill command:
     ```bash
     kill <PID>
     ```
     Where `<PID>` is the process ID number from the previous command

3. Alternative method (stops all processes on port 8000):
   ```bash
   pkill -f "uvicorn api:app"
   ```

You can verify the server has stopped by checking that http://localhost:8000 is no longer accessible.

## API Documentation

Once the server is running, you can access the API documentation at:

- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## API Endpoints

### GET /

Returns a welcome message.

### GET /status

Returns the status of the RAG service.

### POST /query

Query the RAG system with a question.

**Request body:**

```json
{
  "query": "What information does the document contain about...?"
}
```

**Response:**

```json
{
  "answer": "Based on the documents, ...",
  "sources": [
    {
      "file": "document1.pdf",
      "page": 5
    },
    {
      "file": "document2.pdf",
      "page": 12
    }
  ]
}
```

### POST /add-document

Add a new document to the vector store.

**Request body:**

```json
{
  "file_path": "/path/to/document.pdf"
}
```

**Response:**

```json
{
  "status": "success",
  "file": "document.pdf",
  "chunks_added": 15
}
```

## Integration with n8n

To use this API with your n8n workflow:

1. Make sure the API server is running
2. In your n8n workflow, add an HTTP Request node
3. Configure it to send a POST request to `http://localhost:8000/query`
4. Set the request body to include the user's question from WhatsApp
5. Parse the JSON response and send it back to the user

Example n8n HTTP Request node configuration:

- Method: POST
- URL: http://localhost:8000/query
- Body: JSON
- Body content:

```json
{
  "query": "{{$node['Your Previous Node'].json.Body}}"
}
```
