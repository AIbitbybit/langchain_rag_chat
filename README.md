# LangChain RAG Chat Applications

This repository contains a collection of Retrieval-Augmented Generation (RAG) applications built using LangChain.

## Applications

- **Single PDF RAG** (`rag_demos/rag_single_pdf.py`) - A RAG application that:

  - Ingests a PDF document (rental contract)
  - Splits it into meaningful chunks
  - Embeds the chunks using OpenAI's text-embedding-3-small model
  - Stores the embeddings in a Chroma vector database
  - Provides a question-answering interface that retrieves relevant document chunks and uses ChatGPT to generate contextual answers

- **Multiple PDFs RAG** (`rag_demos/rag_collection_of_pdfs.py`) - An enhanced RAG application that:

  - Processes all PDF files from a directory
  - Preserves document source and page information in metadata
  - Splits each document into chunks and combines them into a single vector store
  - Provides contextualized responses that cite the specific source documents and pages
  - Enables querying across multiple documents simultaneously

- **RAG API Service** (`api/api.py` and `api/rag_service.py`) - A FastAPI-based service that:
  - Provides RESTful API endpoints for RAG functionality
  - Allows querying documents through HTTP requests
  - Supports adding new PDF documents to the knowledge base
  - Returns answers with source citations in JSON format
  - Includes Swagger documentation for easy API exploration
  - Can be run with `python run.py` and accessed at http://localhost:8000

For more details on the API implementation, see the [API README](API_README.md).

## Setup

1. Install dependencies:

   ```bash
   poetry install
   ```

2. Create a `.env` file with your OpenAI API key:

   ```
   OPENAI_API_KEY=your_api_key
   ```

3. Place your PDF documents in the `data/` directory

## Usage

Run one of the applications from the repository root:

For single PDF:

```bash
python rag_demos/rag_single_pdf.py
```

For multiple PDFs:

```bash
python rag_demos/rag_collection_of_pdfs.py
```

For the API service:

```bash
python run.py
```

Follow the prompts to ask questions about your documents, or use the API endpoints at http://localhost:8000.
