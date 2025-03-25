import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_service import RAGService

# Load environment variables
load_dotenv()

# Configuration - Updated paths to go one level up
PDF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "chroma_db_api"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(
    title="RAG API Service",
    description="API for RAG (Retrieval-Augmented Generation) using LangChain",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service = None


def get_rag_service():
    global rag_service
    if rag_service is None:
        try:
            rag_service = RAGService(
                pdf_directory=PDF_DIR,
                db_directory=DB_DIR,
                openai_api_key=OPENAI_API_KEY,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize RAG service: {str(e)}"
            )
    return rag_service


# Request and response models
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


class DocumentAddRequest(BaseModel):
    file_path: str


class DocumentAddResponse(BaseModel):
    status: str
    file: str
    chunks_added: int


# API routes
@app.get("/")
async def root():
    return {"message": "Welcome to the RAG API Service"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, rag: RAGService = Depends(get_rag_service)):
    """
    Query the RAG system with a question.

    - **query**: The question to ask
    """
    try:
        result = rag.query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.post("/add-document", response_model=DocumentAddResponse)
async def add_document(
    request: DocumentAddRequest, rag: RAGService = Depends(get_rag_service)
):
    """
    Add a new document to the vector store.

    - **file_path**: Path to the PDF file to add
    """
    try:
        result = rag.add_document(request.file_path)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")


@app.get("/status")
async def status(rag: RAGService = Depends(get_rag_service)):
    """Get the status of the RAG service"""
    return {"status": "operational", "pdf_directory": PDF_DIR, "db_directory": DB_DIR}


# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
