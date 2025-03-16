import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .retrieve.vector_store import ChromaVectorStore

app = FastAPI(
    title="Ollama RAG API",
    description="Retrieval-Augmented Generation API using ChromaDB & Ollama.",
)
vector_store = ChromaVectorStore(db_path="./chroma_db")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    model: str = "mistral"


def query_ollama(query: str, top_k: int, model: str) -> str:
    """
    Queries ChromaDB for relevant context and sends it to Ollama for response generation.

    Args:
        query (str): The user's query.
        top_k (int): Number of relevant documents to retrieve.
        model (str): The Ollama model to use.

    Returns:
        str: The AI-generated response.
    """
    results = vector_store.query(query, top_k=top_k)

    if not results:
        return "No relevant documents found."

    # Build context for Ollama and create the prompt
    context = "\n\n".join([doc["text"] for doc in results])
    prompt = f"""Use the following context to answer the query:

    Context:
    {context}

    Query: {query}
    Answer:"""
    response = ollama.generate(model=model, prompt=prompt)

    return response.get("response", "No response from model.")


@app.post("/ask", summary="Ask a question using the RAG pipeline")
async def ask_question(request: QueryRequest):
    """
    Handles user queries, retrieves relevant documents, and generates responses using Ollama.

    Args:
        request (QueryRequest): Query, top_k results, and model selection.

    Returns:
        dict: User query and AI-generated response.
    """
    response = query_ollama(request.query, request.top_k, request.model)
    return {"query": request.query, "response": response}


@app.get("/", summary="Health check")
async def root():
    """Health check route"""
    return {"message": "Ollama RAG API is running!"}
