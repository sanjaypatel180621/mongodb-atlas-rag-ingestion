import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from create_metadata_tagger import create_metadata_tagger
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from pymongo import MongoClient

load_dotenv('/app/.env')

voyageai_model = "voyage-3.5-lite"
openai_model = "gpt-4o-mini"

mock_voyageai_key = "mock_voyageai_key"
mock_openai_key = "mock_openai_key"

# Initialize the MongoDB client for storing the chunked data
client = MongoClient(os.getenv("CONNECTION_STRING"))
collection = client["langchain_demo"]["chunked_data"]

# Drop the database before adding new data
print("Deleting the collection before adding new data")
collection.delete_many({})

loader = PyPDFLoader("/lab/mongodb.pdf")
pages = loader.load()
cleaned_pages = []

for page in pages:
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)

print("Splitting the documents into chunks")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

schema = {
    "properties": {
        "title": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "hasCode": {"type": "boolean"},
    },
    "required": ["title", "keywords", "hasCode"],
}

print("Creating metadata for the documents")
llm = ChatOpenAI(openai_api_key=mock_openai_key, temperature=0, model=openai_model)
document_transformer = create_metadata_tagger(schema, llm)
docs = document_transformer.transform_documents(cleaned_pages)
split_docs = text_splitter.split_documents(docs)

# Generate an instance of VoyageAIEmbeddings
print("Generating an instance of VoyageAIEmbeddings")
embedding = VoyageAIEmbeddings(model=voyageai_model, voyage_api_key=mock_voyageai_key)

# Store the vectors in MongoDB Atlas
print("Storing the vectors in MongoDB Atlas")
vector_store = MongoDBAtlasVectorSearch.from_documents(
    split_docs, embedding, collection=collection
) 

document_count = collection.count_documents({})
print(f"Successfully stored {document_count} documents in MongoDB Atlas")
client.close()