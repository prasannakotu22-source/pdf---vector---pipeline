import fitz
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("pdf_data")

pdf_path = "document.pdf"

doc = fitz.open(pdf_path)

chunk_id = 0

for page_num in range(len(doc)):
    page = doc[page_num]
text = page.get_text()

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    for chunk in chunks:
        embedding = model.encode(chunk).tolist()

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"page": page_num + 1}],
            ids=[str(chunk_id)]
        )

        chunk_id += 1
      print("PDF ingestion completed")
