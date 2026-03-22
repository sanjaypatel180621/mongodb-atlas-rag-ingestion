# Import the necessary modules
from dotenv import load_dotenv
import os
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings
from langchain_openai import ChatOpenAI

# Load environment variables (ensure your .env has CONNECTION_STRING, OPENAI_API_KEY, and VOYAGE_API_KEY)
load_dotenv("/app/.env")

# --- Configuration ---
voyageai_model = "voyage-3.5-lite"
mock_voyageai_key = os.getenv("VOYAGE_API_KEY")

db_name = "langchain_demo"
collection_name = "chunked_data"
index = "vector_index"
connection_string = os.getenv("CONNECTION_STRING")

# --- Step 1: Create the Vector Store ---
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=connection_string,
    namespace=f"{db_name}.{collection_name}",
    embedding=VoyageAIEmbeddings(model=voyageai_model, voyage_api_key=mock_voyageai_key),
    index_name=index,
)

# --- Step 2: Define the Retriever Function ---
def query_data(query):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}, # Fetching the top 3 most relevant documents
    )
    results = retriever.invoke(query)
    return results

# --- Step 3: Define the Answer Generator ---
def generate_answer(query, retrieved_docs):
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Combine the retrieved document content into a single context string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Build the prompt
    prompt = f"""
    Answer the question based only on the context below:
    Context:
    {context}
    
    Question:
    {query}
    """

    # Generate and return the response
    response = llm.invoke(prompt)
    return response.content

# --- Step 4: End-to-End Execution ---
if __name__ == "__main__":
    query = "When did MongoDB begin supporting multi-document transactions?"
    
    print("Fetching documents...")
    docs = query_data(query)
    
    print("Generating answer...\n")
    answer = generate_answer(query, docs)
    
    print("--- Final Answer ---")
    print(answer)