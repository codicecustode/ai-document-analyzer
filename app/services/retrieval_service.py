"""Retrieval service for vector search operations."""
import logging
from typing import List, Dict
from app.services.vector_db_service import get_pinecone_client
from app.services.embedding_service import get_embedding_model
from app.config.mongodb_config import get_db

logger = logging.getLogger(__name__)


def search_child_chunks(
    query_embedding: List[float],
    index_name: str = "doc-analyzer-child-text",
    top_k: int = 3
) -> List[Dict]:
    """
    Search for similar child chunks using vector similarity.
    
    Args:
        query_embedding: Query vector embedding
        index_name: Name of the Pinecone index
        top_k: Number of results to return
    
    Returns:
        List of dictionaries with "parent_id" and "child_text"
    """
    try:
        pc = get_pinecone_client()
        index = pc.Index(index_name)
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                "parent_id": match.metadata.get("parent_id"),
                "child_text": match.metadata.get("text", ""),
                "score": match.score
            }
            for match in results.matches
        ]
    except Exception as e:
        logger.error(f"Error searching child chunks: {e}")
        raise RuntimeError(f"Vector search failed: {e}") from e