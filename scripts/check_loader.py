
from app.rag.loader import load_documents

documents = load_documents()

print("Total documents:", len(documents))

for doc in documents:
    print(doc["source"], "-", len(doc["content"]), "characters")
