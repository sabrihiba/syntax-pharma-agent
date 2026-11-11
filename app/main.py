
from fastapi import FastAPI

app = FastAPI(
    title="Pharma Research Assistant",
    description="Agentic AI assistant for pharmacology research",
    version="1.0.0"
)


@app.get("/health") #This is a Python decorator. It registers the function below as the handler for HTTP GET requests to /health
def health_check():
    return {"status": "ok"}
