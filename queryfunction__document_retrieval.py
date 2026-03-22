from dotenv import load_dotenv
import os
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings

load_dotenv("/app/.env")
voyageai_model = "voyage-3.5-lite"
mock_voyageai_key = os.getenv("VOYAGE_API_KEY")

db_name = "langchain_demo"
collection_name = "chunked_data"
index = "vector_index"
connection_string = os.getenv("CONNECTION_STRING")

# TODO: Complete the vectorStore definition by providing the connection string, db_name.collection, VoyageAIEmbeddings instance, and index_name
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=connection_string,
    namespace=f"{db_name}.{collection_name}",
    embedding=VoyageAIEmbeddings( model=voyageai_model, voyage_api_key=mock_voyageai_key),
    index_name=index,
)


# TODO: Complete the query_data function
def query_data(query):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    results = retriever.invoke(query)
    return results

# Query the data
print(query_data("When did MongoDB begin supporting multi-document transactions?"))