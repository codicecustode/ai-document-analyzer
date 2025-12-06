"""Embedding service for generating text embeddings."""
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import Optional

# Singleton embedding model instance
_embedding_model: Optional[GoogleGenerativeAIEmbeddings] = None


def get_embedding_model(model_name: str = "models/gemini-embedding-001") -> GoogleGenerativeAIEmbeddings:
    """
    Get or create embedding model (singleton pattern).
    
    Args:
        model_name: Name of the embedding model
    
    Returns:
        GoogleGenerativeAIEmbeddings: Initialized embedding model
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)
    return _embedding_model
