import os
from typing import Any, Callable, Dict, List, Optional

import chromadb
import ollama
from sentence_transformers import SentenceTransformer


class ChromaVectorStore:
    """
    Handles storage and retrieval of chunked text embeddings using ChromaDB.
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        custom_embedder: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        """
        Initializes ChromaDB client and sets up embedding method.

        Args:
            db_path (str): Path to store ChromaDB files.
            embedding_model (str): Either an Ollama model (e.g., "ollama:mxbai-embed-large")
                                  or a Sentence Transformer model (e.g., "sentence-transformers/all-MiniLM-L6-v2").
            custom_embedder (Optional[Callable]): A function that takes a list of texts and returns embeddings.
                                                  If provided, this overrides the default model.
        """
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="documents")
        self.embedding_model = embedding_model
        self.custom_embedder = custom_embedder

        # Set up embedding model
        if custom_embedder:
            self.embedder = custom_embedder
        elif embedding_model.startswith("ollama"):
            self.embedder = self._embed_with_ollama
        else:
            self.model = SentenceTransformer(embedding_model)
            self.embedder = self._embed_with_transformers

    def _embed_with_ollama(self, texts: List[str]) -> List[List[float]]:
        """Uses Ollama's embedding model to convert text into vector embeddings."""
        response = ollama.embed(model=self.embedding_model.split(":")[1], input=texts)
        return response["embeddings"]

    def _embed_with_transformers(self, texts: List[str]) -> List[List[float]]:
        """Uses SentenceTransformers to generate embeddings and converts them to lists."""
        return self.model.encode(texts).tolist()

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Embeds and adds documents to ChromaDB.

        Args:
            chunks (List[Dict[str, Any]]): List of chunked documents with metadata.
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder(texts)  # Use selected embedding method

        # Generate unique IDs for each chunk
        ids = [
            f"{chunk['metadata']['file_path']}_{chunk['metadata']['row_index']}_{chunk['metadata']['chunk_index']}"
            for chunk in chunks
        ]

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[chunk["metadata"] for chunk in chunks],
        )

        print(f"Added {len(chunks)} documents to ChromaDB.")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Queries ChromaDB for relevant documents based on similarity.

        Args:
            query_text (str): The user query.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict]: Retrieved documents with metadata.
        """
        query_embedding = self.embedder([query_text])

        results = self.collection.query(
            query_embeddings=query_embedding, n_results=top_k
        )

        retrieved_docs = []
        for i in range(len(results["documents"][0])):
            retrieved_docs.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return retrieved_docs


if __name__ == "__main__":

    # Create a sample dataset
    sample_chunks = [
        {
            "text": "Dry the towels on the rack, not on the floor.",
            "metadata": {
                "file_path": "sample.csv",
                "row_index": 0,
                "chunk_index": 0,
                "author": "John Doe",
            },
        },
        {
            "text": "Llamas were first domesticated 4,000 years ago.",
            "metadata": {
                "file_path": "sample.csv",
                "row_index": 1,
                "chunk_index": 0,
                "author": "Jane Smith",
            },
        },
        {
            "text": "Llamas are mammals often found in South America.",
            "metadata": {
                "file_path": "sample.csv",
                "row_index": 2,
                "chunk_index": 0,
                "author": "Jane Smith",
            },
        },
    ]

    vector_store = ChromaVectorStore(db_path="./chroma_db")

    # Add sample documents
    print("\nAdding documents to ChromaDB...")
    vector_store.add_documents(sample_chunks)

    # Query ChromaDB
    query_text = "What is a llama?"
    print(f"\nQuerying for: '{query_text}'")
    results = vector_store.query(query_text, top_k=3)

    print("\nQuery Results:")
    for i, res in enumerate(results):
        print(f"Result {i+1}:")
        print("Text:", res["text"])
        print("Metadata:", res["metadata"])
        print("Distance:", res["distance"])
        print("-" * 40)
