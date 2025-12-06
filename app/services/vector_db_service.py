"""Vector database service for Pinecone operations."""
import os
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from typing import Optional

logger = logging.getLogger(__name__)

# Singleton Pinecone client
_pinecone_client: Optional[Pinecone] = None


def get_pinecone_client() -> Pinecone:
    """
    Get or create Pinecone client (singleton pattern).
    
    Returns:
        Pinecone: Initialized Pinecone client
    
    Raises:
        ValueError: If PINECONE_API_KEY is not set
    """
    global _pinecone_client
    if _pinecone_client is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        _pinecone_client = Pinecone(api_key=api_key)
    return _pinecone_client


def create_index_if_not_exists(
    index_name: str,
    dimension: int = 3072,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1"
) -> None:
    """
    Create Pinecone index if it doesn't exist.
    
    Args:
        index_name: Name of the index
        dimension: Vector dimension
        metric: Similarity metric
        cloud: Cloud provider
        region: AWS region
    """
    pc = get_pinecone_client()
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
        logger.info(f"Created Pinecone index: {index_name}")
    else:
        logger.debug(f"Pinecone index already exists: {index_name}")


def get_vector_store(
    index_name: str,
    embeddings,
    create_if_not_exists: bool = True
) -> PineconeVectorStore:
    """
    Get or create a Pinecone vector store.
    
    Args:
        index_name: Name of the Pinecone index
        embeddings: Embedding model instance
        create_if_not_exists: Whether to create index if it doesn't exist
    
    Returns:
        PineconeVectorStore: Initialized vector store
    """
    pc = get_pinecone_client()
    
    if create_if_not_exists:
        create_index_if_not_exists(index_name)
    
    index = pc.Index(index_name)
    return PineconeVectorStore(index=index, embedding=embeddings)


def add_documents_to_vector_store(vector_store: PineconeVectorStore, documents: list) -> None:
    """
    Add documents to a vector store.
    
    Args:
        vector_store: PineconeVectorStore instance
        documents: List of Document objects to add
    """
    try:
        vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to vector store")
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {e}")
        raise
