import glob
import os  # noqa

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


# Create embeddings
print("\n--- Creating embeddings ---")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def init_vectorstore(pdf_directory, db_dir):
    # Find all PDF files in the directory
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_directory}")

    print(f"\n--- Found {len(pdf_files)} PDF files ---")

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

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Total number of document chunks: {len(all_docs)}")
    print(f"Sample chunk from: {all_docs[0].metadata.get('source_file', 'Unknown')}")
    print(f"Content: {all_docs[0].page_content[:200]}...\n")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    Chroma.from_documents(all_docs, embeddings, persist_directory=db_dir)
    print("\n--- Finished creating vector store ---")


def query_vectorstore(query, db_dir):
    # Load the existing vector store with the embedding function
    db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},  # Increased to 5 for multiple documents
    )
    relevant_docs = retriever.invoke(query)

    # Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        print(f"Document {i}: [Source: {source}, Page: {page}]")

    # Combine the query and the relevant document contents
    combined_input = (
        "Here are some documents that might help answer the question: "
        + query
        + "\n\nRelevant Documents:\n"
    )

    # Add each document with its source information
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        combined_input += (
            f"\n\nDocument {i} [Source: {source}, Page: {page}]:\n{doc.page_content}"
        )

    combined_input += "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'. Always mention which document(s) and page(s) you used for your answer."

    # Create a ChatOpenAI model
    model = ChatOpenAI(model="gpt-4o")

    # Define the messages for the model
    messages = [
        SystemMessage(
            content="You are a helpful assistant. Answer based only on the provided documents."
        ),
        HumanMessage(content=combined_input),
    ]

    # Invoke the model with the combined input
    result = model.invoke(messages)

    # Display the full result and content only
    print("\n--- Generated Response ---")
    print("Content only:")
    print(result.content)


def main():
    # Define the directory containing the PDF files and the persistent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    pdf_directory = os.path.join(parent_dir, "data")
    persistent_directory = os.path.join(parent_dir, "db", "chroma_db_multiple_pdfs")

    # Check if the Chroma vector store already exists
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")
        init_vectorstore(pdf_directory, persistent_directory)

    query = input("Add your question here: ")
    query_vectorstore(query, persistent_directory)


if __name__ == "__main__":
    main()
