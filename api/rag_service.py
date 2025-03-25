import glob
import os
from typing import Any, Dict, List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RAGService:
    """Service for handling RAG (Retrieval-Augmented Generation) operations."""

    def __init__(
        self, pdf_directory: str, db_directory: str, openai_api_key: str = None
    ):
        """
        Initialize the RAG service.

        Args:
            pdf_directory: Directory containing PDF files to index
            db_directory: Directory to store the vector database
            openai_api_key: OpenAI API key (optional if set in environment)
        """
        # Set OpenAI API key if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        self.pdf_directory = pdf_directory
        self.db_directory = db_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o")

        # Initialize vector store if it doesn't exist
        if not os.path.exists(db_directory):
            print("Initializing vector store...")
            self._init_vectorstore()
        else:
            print(f"Using existing vector store at {db_directory}")

        # Load the vector store
        self.db = Chroma(
            persist_directory=db_directory, embedding_function=self.embeddings
        )
        self.retriever = self.db.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

    def _init_vectorstore(self) -> None:
        """Initialize the vector store from PDF files."""
        # Find all PDF files in the directory
        pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.pdf_directory}")

        print(f"Found {len(pdf_files)} PDF files")

        all_docs = []

        # Process each PDF file
        for pdf_file in pdf_files:
            print(f"Processing: {os.path.basename(pdf_file)}")
            # Read the text content from the file
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()

            # Add source information to metadata
            for doc in documents:
                doc.metadata["source_file"] = os.path.basename(pdf_file)

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            docs = text_splitter.split_documents(documents)

            print(f"  - Generated {len(docs)} chunks")
            all_docs.extend(docs)

        print(f"Total number of document chunks: {len(all_docs)}")

        # Create the vector store and persist it automatically
        Chroma.from_documents(
            all_docs, self.embeddings, persist_directory=self.db_directory
        )
        print("Finished creating vector store")

    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.

        Args:
            query_text: The question to answer

        Returns:
            Dict containing the answer and source information
        """
        # Retrieve relevant documents
        relevant_docs = self.retriever.invoke(query_text)

        # Prepare source information
        sources = []
        for i, doc in enumerate(relevant_docs):
            source = {
                "file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
            }
            sources.append(source)

        # Create the prompt with all relevant documents
        combined_input = (
            "Here are some documents that might help answer the question: "
            + query_text
            + "\n\nRelevant Documents:\n"
        )

        # Add each document with its source information
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            combined_input += f"\n\nDocument {i} [Source: {source}, Page: {page}]:\n{doc.page_content}"

        combined_input += "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'. Do not refer to the documents or make reference to them"

        # Define the messages for the model
        messages = [
            SystemMessage(
                content="You are a helpful assistant. Answer based only on the provided documents."
            ),
            HumanMessage(content=combined_input),
        ]

        # Invoke the model with the combined input
        result = self.llm.invoke(messages)

        # Return the result
        return {"answer": result.content, "sources": sources}

    def add_document(self, file_path: str) -> Dict[str, Any]:
        """
        Add a new document to the vector store.

        Args:
            file_path: Path to the PDF file to add

        Returns:
            Dict containing status and number of chunks added
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.endswith(".pdf"):
            raise ValueError("Only PDF files are supported")

        # Load the document
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Add source information
        for doc in documents:
            doc.metadata["source_file"] = os.path.basename(file_path)

        # Split the document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

        # Add to the vector store
        self.db.add_documents(docs)
        self.db.persist()

        return {
            "status": "success",
            "file": os.path.basename(file_path),
            "chunks_added": len(docs),
        }
