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


def init_vectorstore(file_path, db_dir):

    # Read the text content from the file
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    Chroma.from_documents(docs, embeddings, persist_directory=db_dir)

    print("\n--- Finished creating vector store ---")


def query_vectorstore(query, db_dir):
    # Load the existing vector store with the embedding function
    db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    relevant_docs = retriever.invoke(query)

    # Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    # for i, doc in enumerate(relevant_docs, 1):
    #     print(f"Document {i}:\n{doc.page_content}\n")
    #     if doc.metadata:
    #         print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

    # Combine the query and the relevant document contents
    combined_input = (
        "Here are some documents that might help answer the question: "
        + query
        + "\n\nRelevant Documents:\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
    )

    # Create a ChatOpenAI model
    model = ChatOpenAI(model="gpt-4o")

    # Define the messages for the model
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    # Invoke the model with the combined input
    result = model.invoke(messages)

    # Display the full result and content only
    print("\n--- Generated Response ---")
    # print("Full result:")
    # print(result)
    print("Content only:")
    print(result.content)


def main():

    # Define the directory containing the text file and the persistent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, "data", "my_rental_contract.pdf")
    persistent_directory = os.path.join(parent_dir, "db", "chroma_db_rental_contract")

    # Check if the Chroma vector store already exists
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The file {file_path} does not exist. Please check the path."
            )
        init_vectorstore(file_path, persistent_directory)

    query = input("Add your question here: ")
    query_vectorstore(query, persistent_directory)


if __name__ == "__main__":
    main()
