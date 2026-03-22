# mongodb-atlas-rag-ingestion
A complete ingestion pipeline for a Retrieval-Augmented Generation (RAG) application using MongoDB Atlas Vector Search, LangChain, OpenAI, and VoyageAI.

# Creating a RAG Application with MongoDB Atlas

This repository demonstrates the document ingestion phase of building a Retrieval-Augmented Generation (RAG) application. It takes a source PDF document, chunks the text, generates vector embeddings using VoyageAI, extracts metadata using OpenAI, and stores the final document vectors in MongoDB Atlas for efficient similarity search.

## Overview

The `load_data.py` script performs the following operations:
1. **Connects to MongoDB:** Initializes a connection to a MongoDB database (`langchain_demo`) and collection (`chunked_data`), clearing any existing data.
2. **Loads & Filters Data:** Reads `mongodb.pdf` using LangChain's `PyPDFLoader` and removes pages with less than 20 words.
3. **Chunks Text:** Splits the documents into smaller, manageable chunks using `RecursiveCharacterTextSplitter`.
4. **Tags Metadata:** Uses an OpenAI LLM (`gpt-4o-mini`) to automatically generate metadata (title, keywords, and a boolean flag indicating if the text contains code).
5. **Generates Embeddings:** Converts the text chunks into vector embeddings using the `VoyageAIEmbeddings` model (`voyage-3.5-lite`).
6. **Stores in Atlas:** Loads the text, metadata, and embeddings into MongoDB Atlas using `MongoDBAtlasVectorSearch`.

## Prerequisites

To run this experiment, you need:
* Python 3.x
* A MongoDB Atlas Cluster (or local instance) with a Vector Search index configured.
* Environment variables set in a `.env` file for your connection string and API keys. *(Note: This lab uses mock API keys for demonstration purposes).*

## Running the Script

Execute the script from your terminal:

```bash
python3 load_data.py
```

## Expected Terminal Output
Plaintext
Deleting the collection before adding new data
Splitting the documents into chunks
Creating metadata for the documents
Generating an instance of VoyageAIEmbeddings
Storing the vectors in MongoDB Atlas
Successfully stored 170 documents in MongoDB Atlas
Verifying the Data in MongoDB
You can verify that the vector embeddings and metadata were successfully stored by connecting to your database via mongosh and running a findOne() query.

## JavaScript

// Connect to your database
use langchain_demo

// Query the collection
db.chunked_data.findOne()
Expected Database Document
Note: The embedding array has been truncated for readability.

```json
JSON
{
  "_id": "ObjectId('69bf7348cb2ac8de90c8ac96')",
  "text": "About This Book\nLicense\nThe Little MongoDB Book is licensed under the Attribution-NonCommercial 3.0 Unported\nlicense. Y ou should not have paid for this book.\nYou are basically free to copy, distribute, modify or display the book. However, please\nalways attribute the book to its original author - Karl Seguin - and do not use it for com-\nmercial purposes. You can see the full text of the license at:http://creativecommons.\norg/licenses/by-nc/3.0/legalcode\nAbout The Original Author",
  "embedding": [
      -0.019175313413143158, 0.02580602839589119, -0.005667469464242458,
      "... (1536 dimensions total) ..."
  ],
  "title": "The Little MongoDB Book",
  "keywords": [ "license", "author", "MongoDB Inc", "latest version" ],
  "hasCode": false,
  "producer": "xdvipdfmx (20190503)",
  "creator": "LaTeX with hyperref",
  "creationdate": "2022-05-27T16:48:41-04:00",
  "source": "/lab/mongodb.pdf",
  "total_pages": 36,
  "page": 1,
  "page_label": "1"
}
```


### `load_data.py`
Here is your finalized python file ready to be committed to the repository:

```python
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
```


