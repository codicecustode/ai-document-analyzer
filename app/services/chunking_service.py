"""Text chunking service for hierarchical document splitting."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Tuple, List, Dict


def get_text_splitters(
    parent_size: int = 1500,
    parent_overlap: int = 200,
    child_size: int = 500,
    child_overlap: int = 100
) -> Tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]:
    """
    Get parent and child text splitters for hierarchical chunking.
    
    Args:
        parent_size: Size of parent chunks
        parent_overlap: Overlap between parent chunks
        child_size: Size of child chunks
        child_overlap: Overlap between child chunks
    
    Returns:
        Tuple of (parent_splitter, child_splitter)
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
    )
    return parent_splitter, child_splitter


def chunk_text_hierarchically(
    document_text: str,
    parent_splitter: RecursiveCharacterTextSplitter,
    child_splitter: RecursiveCharacterTextSplitter
) -> Tuple[List[Dict], List[Document]]:
    """
    Split document into hierarchical parent and child chunks.
    
    Args:
        document_text: Full document text to chunk
        parent_splitter: Splitter for parent chunks
        child_splitter: Splitter for child chunks
    
    Returns:
        Tuple of (parent_docs, child_docs)
        - parent_docs: List of dicts with "text" and "parent_id"
        - child_docs: List of Document objects with metadata
    """
    # Create parent chunks
    parent_chunks = parent_splitter.split_text(document_text)
    
    # Create child chunks from each parent chunk
    child_docs = []
    for parent_idx, parent_chunk in enumerate(parent_chunks):
        child_chunks = child_splitter.split_text(parent_chunk)
        for child_chunk in child_chunks:
            child_docs.append(Document(
                page_content=child_chunk,
                metadata={"parent_id": parent_idx}
            ))
    
    # Format parent chunks as dicts
    parent_docs = [
        {
            "text": text,
            "parent_id": idx
        }
        for idx, text in enumerate(parent_chunks)
    ]
    
    return parent_docs, child_docs
