# LangChain RAG Chat Applications

This repository contains a collection of Retrieval-Augmented Generation (RAG) applications built using LangChain.

## Applications

- **Single PDF RAG** (`rag_demos/rag_single_pdf.py`) - A RAG application that:
  - Ingests a PDF document (rental contract)
  - Splits it into meaningful chunks
  - Embeds the chunks using OpenAI's text-embedding-3-small model
  - Stores the embeddings in a Chroma vector database
  - Provides a question-answering interface that retrieves relevant document chunks and uses ChatGPT to generate contextual answers

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

```bash
python rag_demos/rag_single_pdf.py
```

Follow the prompts to ask questions about your document.
