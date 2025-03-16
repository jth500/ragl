from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import ollama
from sentence_transformers import SentenceTransformer


class BaseEmbedder(ABC):
    """
    Abstract base class for embedding models.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of text inputs into vector embeddings.

        Args:
            texts (List[str]): A list of text documents.

        Returns:
            List[List[float]]: Corresponding list of vector embeddings.
        """
        pass


class OllamaEmbedder(BaseEmbedder):
    """
    Uses Ollama's embedding models for text embeddings.
    """

    def __init__(self, model: str = "mxbai-embed-large"):
        """
        Initializes Ollama-based embedder.

        Args:
            model (str): Ollama embedding model name. Defaults to "mxbai-embed-large".
        """
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings using Ollama."""
        response = ollama.embed(model=self.model, input=texts)
        return response["embeddings"]


class SentenceTransformersEmbedder(BaseEmbedder):
    """
    Uses SentenceTransformers for text embeddings.
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes SentenceTransformers-based embedder.

        Args:
            model (str): Sentence Transformers model name. Defaults to "all-MiniLM-L6-v2".
        """
        self.model = SentenceTransformer(model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings using Sentence Transformers."""
        return self.model.encode(texts).tolist()
