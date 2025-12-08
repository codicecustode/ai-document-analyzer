"""MongoDB service for document storage operations."""
import logging
from app.config.mongodb_config import get_db
from typing import List, Dict

logger = logging.getLogger(__name__)


async def save_parent_chunks(chunks: List[Dict]) -> None:
    """
    Save parent document chunks to MongoDB.
    
    Args:
        chunks: List of parent chunk dictionaries with "text" and "parent_id"
    
    Raises:
        Exception: If database operation fails
    """
    try:
        db = await get_db()
        result = await db["parent_doc_chunk_collection"].insert_many(chunks)
        logger.info(f"Saved {len(result.inserted_ids)} parent chunks to MongoDB")
    except Exception as e:
        logger.error(f"Error saving parent chunks to MongoDB: {e}")
        raise


async def get_parent_chunks_by_ids(parent_ids: List[int]) -> List[Dict]:
    """
    Retrieve parent chunks by their IDs.
    
    Args:
        parent_ids: List of parent_id integers
    
    Returns:
        List of parent chunk documents
    """
    try:
        db = await get_db()
        cursor = db["parent_doc_chunk_collection"].find(
            {"parent_id": {"$in": parent_ids}}
        )
        results = await cursor.to_list(length=None)
        return results
    except Exception as e:
        logger.error(f"Error retrieving parent chunks: {e}")
        raise

