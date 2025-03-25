#!/usr/bin/env python
"""
Main run script for the RAG API service.
This script simplifies starting the API from the project root.
"""

import os
import sys

import uvicorn

# Add the api directory to the system path
api_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api")
sys.path.insert(0, api_path)

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn, referencing the module in the api directory
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
