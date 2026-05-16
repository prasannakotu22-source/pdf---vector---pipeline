from fastapi import FastAPI
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("pdf_collection")

model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/")
def home():
    return {"message": "PDF Vector Pipeline Running"}

@app.get("/search/")
def search(query: str):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return {
        "results": results["documents"]
    }
