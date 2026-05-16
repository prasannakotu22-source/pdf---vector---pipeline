from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import chromadb

app = FastAPI()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("pdf_chunks")

model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/query")
def query_pdf(data: dict):
    query = data["query"]
    top_k = data.get("top_k", 3)

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    output = [] 

    for i in range(len(results["documents"][0])):
        output.append({
            "chunk_text": results["documents"][0][i],
            "page_number": results["metadatas"][0][i]["page"],
            "score": results["distances"][0][i]
        })

    return output
