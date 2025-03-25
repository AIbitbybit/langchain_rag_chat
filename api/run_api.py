#!/usr/bin/env python
"""
Run script for the RAG API server.
"""
import uvicorn


def main():
    """Run the FastAPI app using uvicorn."""
    uvicorn.run(
        "langchain_rag_chat.api.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
