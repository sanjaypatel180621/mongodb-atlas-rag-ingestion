# Mongodb Atlas RAG Ingestion Application using MonogoDB Labs (End-to-End) 
A complete ingestion pipeline for a Retrieval-Augmented Generation (RAG) application using MongoDB Atlas Vector Search, LangChain, OpenAI, and VoyageAI.

This repository demonstrates how to build a complete **Retrieval-Augmented Generation (RAG)** pipeline using:

- MongoDB Atlas Vector Search  
- LangChain  
- OpenAI (LLM for metadata & answer generation)  
- VoyageAI (Embeddings)  

I have worked to covers the **3 core stages of a RAG system**:
1. Data Ingestion & Embedding Generation  
2. Document Retrieval (Retriever)  
3. Answer Generation (Generator)

---

# Problem Statement 1 - Generate Vector Embeddings

This repository where I demonstrates my skills on the document ingestion phase of building a Retrieval-Augmented Generation (RAG) application. It takes a source PDF document, chunks the text, generates vector embeddings using VoyageAI, extracts metadata using OpenAI, and stores the final document vectors in MongoDB Atlas for efficient similarity search.

## Objective
I spend time to learn how to generate vector embeddings from documents and store them in MongoDB Atlas for efficient semantic search.

## Overview - Mongodb Atlas RAG Ingestion

This module focuses on document ingestion, processing, and embedding generation.

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

## Architecture – Vector Embedding Generation

```text
          ┌──────────────────────┐
          │     Source PDF       │
          │   (mongodb.pdf)      │
          └─────────┬────────────┘
                    │
                    ▼
          ┌──────────────────────┐
          │   PyPDFLoader        │
          │ (LangChain Loader)   │
          └─────────┬────────────┘
                    │
                    ▼
          ┌──────────────────────┐
          │   Data Cleaning      │
          │ (Remove small pages) │
          └─────────┬────────────┘
                    │
                    ▼
          ┌────────────────────────────┐
          │ Text Splitter              │
          │ RecursiveCharacterSplitter │
          └─────────┬──────────────────┘
                    │
                    ▼
          ┌────────────────────────────┐
          │ Metadata Generation        │
          │ OpenAI (gpt-4o-mini)       │
          └─────────┬──────────────────┘
                    │
                    ▼
          ┌────────────────────────────┐
          │ Embedding Generation       │
          │ VoyageAI (voyage-3.5-lite) │
          └─────────┬──────────────────┘
                    │
                    ▼
          ┌────────────────────────────┐
          │ MongoDB Atlas              │
          │ Vector Search Collection   │
          └────────────────────────────┘
```
---

# Problem Statement 2 - Implement the Retriever using RAG & Mongodb Atlas

## Overview - Implement the Retriever using RAG & Mongodb Atlas
This repository demonstrates how to implement the retriever component of a Retrieval-Augmented Generation (RAG) system. Using LangChain, VoyageAI for embeddings, and MongoDB Atlas as the vector store, this script retrieves the most relevant document chunks based on a user's natural language query.

## Overview - Mongodb Atlas RAG Ingestion - Implementing a RAG Retriever with MongoDB Atlas

This repository demonstrates how to implement the retriever component of a Retrieval-Augmented Generation (RAG) system. Using LangChain, VoyageAI for embeddings, and MongoDB Atlas as the vector store, this script retrieves the most relevant document chunks based on a user's natural language query.

The `demo.py` script performs the following operations:

* **Connects to Atlas Vector Search:** Initializes a connection to the existing MongoDB database (`langchain_demo`) and collection (`chunked_data`) where our document vectors are stored.
* **Configures the Retriever:** Converts the vector store into a retriever interface, instructing it to perform a "similarity" search and return the top 3 most relevant results (`k: 3`).
* **Executes a Query:** Passes the query *"When did MongoDB begin supporting multi-document transactions?"* to the retriever and prints the resulting document chunks.

## Changes Required to Complete the Lab
To successfully build the retriever from the starter template, the following `TODO` sections were completed:

### 1. Vector Store Definition
The `MongoDBAtlasVectorSearch.from_connection_string` method needed the database routing information.
* **`connection_string`**: Set to the `connection_string` variable.
* **`namespace`**: Formatted using an f-string to combine the database and collection names: `f"{db_name}.{collection_name}"`.
* **`index_name`**: Set to the `index` variable (`"vector_index"`).

### 2. Retriever Configuration
The `vector_store.as_retriever()` method needed search parameters to know how to fetch data.
* **`search_type`**: Set to `"similarity"` to perform a standard vector math comparison.
* **`search_kwargs`**: Set to `{"k": 3}` to limit the returned documents to the top 3 most relevant matches.

## `demo.py` Code
Here is the fully completed script:

```python
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

# Completed the vectorStore definition by providing the connection string, namespace, and index_name
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=connection_string,
    namespace=f"{db_name}.{collection_name}",
    embedding=VoyageAIEmbeddings(model=voyageai_model, voyage_api_key=mock_voyageai_key),
    index_name=index,
)

# Completed the query_data function
def query_data(query):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    results = retriever.invoke(query)
    return results

# Query the data
print(query_data("When did MongoDB begin supporting multi-document transactions?"))
```

### Retriever Function
def query_data(query):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    return retriever.invoke(query)


## Running the Script

Execute the script from your terminal:

```bash
python3 demo.py
```

### Example Query
query_data("When did MongoDB begin supporting multi-document transactions?")

### Output
Returns top 3 most relevant document chunks based on semantic similarity.


### Expected Terminal Output
The script successfully fetches the 3 most relevant text chunks containing the answer.

```Plaintext
[
  {
    "metadata": {
      "_id": "68bee58bc2dfb17fdc2461bc",
      "title": "MongoDB Features and Capabilities",
      "keywords": [
        "capped collections",
        "TTL Indexes",
        "durability",
        "journaling",
        "full text search",
        "transactions",
        "ACID transactions",
        "atomic update operations"
      ],
      "hasCode": false,
      "producer": "xdvipdfmx (20190503)",
      "creator": "LaTeX with hyperref",
      "creationdate": "2022-05-27T16:48:41-04:00",
      "source": "/lab/mongodb.pdf",
      "total_pages": 36,
      "page": 31,
      "page_label": "31"
    },
    "page_content": "engine. With MongoDB’s support for arrays and full text search, you will only need to\nlook to other solutions if you need a more powerful and full-featured full text search\nengine.\nTransactions\nMongoDB added full support for ACID transactions26 in 4.0 (extending it to sharded\nclusters in 4.2). Before that there were two alternatives, one which is great and still has\nits place, and the other that was cumbersome but flexible."
  },
  {
    "metadata": {
      "_id": "68bee58bc2dfb17fdc2461c0",
      "title": "MongoDB Features and Transactions",
      "keywords": [
        "findAndModify",
        "atomicity",
        "two-phase commit",
        "transactions",
        "MongoDB multi-document transactions",
        "retry functionality",
        "geospatial indexes",
        "geoJSON",
        "replication",
        "replica sets",
        "primary",
        "secondary"
      ],
      "hasCode": false,
      "producer": "xdvipdfmx (20190503)",
      "creator": "LaTeX with hyperref",
      "creationdate": "2022-05-27T16:48:41-04:00",
      "source": "/lab/mongodb.pdf",
      "total_pages": 36,
      "page": 32,
      "page_label": "32"
    },
    "page_content": "commit/rollback steps manually. This is the case where using MongoDB multi-document\ntransactions is a great option as they significantly simplify the application code.\nUsing transactions in MongoDB is as straight forward as in relational databases. You\nstart a transaction which later you can commit or abort. To simplify your code even\nfurther, drivers automatically provide retry functionality on retryable errors.\nGeospatial"
  },
  {
    "metadata": {
      "_id": "68bee58bc2dfb17fdc2461bd",
      "title": "MongoDB Features and Capabilities",
      "keywords": [
        "capped collections",
        "TTL Indexes",
        "durability",
        "journaling",
        "full text search",
        "transactions",
        "ACID transactions",
        "atomic update operations"
      ],
      "hasCode": false,
      "producer": "xdvipdfmx (20190503)",
      "creator": "LaTeX with hyperref",
      "creationdate": "2022-05-27T16:48:41-04:00",
      "source": "/lab/mongodb.pdf",
      "total_pages": 36,
      "page": 31,
      "page_label": "31"
    },
    "page_content": "its place, and the other that was cumbersome but flexible.\nThe first is its many atomic update operations. These are great, so long as they actually\naddress your problem. We already saw some of the simpler ones, like$inc and $set.\n25[http://docs.mongodb.org/manual/tutorial/expire-data/](http://docs.mongodb.org/manual/tutorial/expire-data/)\n26[https://www.mongodb.com/docs/manual/core/transactions/](https://www.mongodb.com/docs/manual/core/transactions/)\n31"
  }
]
```

## Architecture – Implement the Retriever using RAG & Mongodb Atlas and Document Retriever (RAG)

```text
           ┌──────────────────────┐
           │     User Query       │
           │ "Natural Language"   │
           └─────────┬────────────┘
                     │
                     ▼
           ┌────────────────────────────┐
           │ Query Embedding Generation │
           │ VoyageAI Embeddings        │
           └─────────┬──────────────────┘
                     │
                     ▼
           ┌────────────────────────────┐
           │ MongoDB Atlas Vector Search│
           │ (Similarity Search)        │
           └─────────┬──────────────────┘
                     │
                     ▼
           ┌────────────────────────────┐
           │ Top-K Documents Retrieved  │
           │ (k = 3 most relevant)      │
           └────────────────────────────┘
```
---

# Problem Statement 3 – Build the Answer Generator

## Objective
Generate accurate answers using retrieved documents and LLM.

## Overview - Generate answers to specific prompts in your RAG application using a custom prompt template and a series of steps
This module completes the RAG pipeline by integrating retrieval with LLM-based answer generation.

### Workflow: 
1 - Accept user query
2 - Retrieve relevant documents
3 - Pass context to LLM
4 - Generate final answer

## Implementation

```python
from langchain_openai import ChatOpenAI

def generate_answer(query, retrieved_docs):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    Answer the question based only on the context below:

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)
    return response.content
```

### End-to-End Example to run it on Labs

```python
query = "When did MongoDB begin supporting multi-document transactions?"

docs = query_data(query)
answer = generate_answer(query, docs)

print(answer)
```

### RAG Architecture Flow
```plaintext
User Query
   ↓
Retriever (MongoDB Vector Search)
   ↓
Top-K Documents
   ↓
LLM (OpenAI)
   ↓
Final Answer
```

## Architecture – Answer Generator (RAG)

```text
           ┌──────────────────────┐
           │     User Query       │
           └─────────┬────────────┘
                     │
                     ▼
           ┌────────────────────────────┐
           │ Retriever (Problem 2)      │
           │ Fetch Top-K Documents      │
           └─────────┬──────────────────┘
                     │
                     ▼
           ┌────────────────────────────┐
           │ Context Builder            │
           │ Combine Retrieved Docs     │
           └─────────┬──────────────────┘
                     │
                     ▼
           ┌────────────────────────────┐
           │ Prompt Template            │
           │ (Context + Question)       │
           └─────────┬──────────────────┘
                     │
                     ▼
           ┌────────────────────────────┐
           │ LLM (OpenAI GPT-4o-mini)   │
           │ Answer Generation          │
           └─────────┬──────────────────┘
                     │
                     ▼
           ┌──────────────────────┐
           │   Final Answer       │
           └──────────────────────┘
```
---

## Prerequisites

Ensure you have the following before running this project:

- Python 3.x  
- MongoDB Atlas Cluster with Vector Search Index configured  
- API Keys:
  - OpenAI API Key  
  - VoyageAI API Key  

---

## Environment Variables (.env)

Create a `.env` file in the root directory and add:

```env
CONNECTION_STRING=your_mongodb_connection
OPENAI_API_KEY=your_openai_key
VOYAGE_API_KEY=your_voyage_key
```

## Labs Summery
| Component | Description                                                      |
| --------- | ---------------------------------------------------------------- |
| Ingestion | Load, chunk, generate embeddings, and store documents in MongoDB |
| Retriever | Fetch relevant document chunks using vector similarity search    |
| Generator | Generate accurate answers using LLM with retrieved context       |


---
**Brief Explanation about what we did in MongoDB Labs:**
- First we convert documents into embeddings and store in vector DB  
- Then user query is converted into embedding and matched using similarity search  
- Finally, retrieved context is passed to LLM to generate accurate answer  

**One-line Summary:**
> *“RAG = Retriever (MongoDB Vector Search) + Generator (LLM)”*

### Key Learnings
Build end-to-end Retrieval-Augmented Generation (RAG) applications
Implement vector search using MongoDB Atlas
Design semantic search systems using embeddings
Integrate LLMs with real-world datasets for intelligent responses

## Author

<p align="left">
  <img src="lab-img.png" alt="Author Image" width="150"/>
</p>

<p align="left">
  <b><a href="https://www.google.com/search?q=sanjay+chintamani+patel">Sanjay Chintamani Patel</a></b><br/>
  Full Stack AI Architect (TCS)
</p>
