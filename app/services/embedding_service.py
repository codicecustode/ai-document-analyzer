"""Embedding service using Google Generative AI."""
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import Optional

_embedding_model: Optional[GoogleGenerativeAIEmbeddings] = None


def get_embedding_model(model_name: str = "models/gemini-embedding-001") -> GoogleGenerativeAIEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for embeddings")
        _embedding_model = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
        )
    return _embedding_model
