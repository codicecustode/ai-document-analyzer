"""Document metadata service for storing and retrieving document information using file storage."""
import logging
import json
import os
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Metadata storage directory
METADATA_DIR = Path(__file__).parent.parent / "metadata"
METADATA_DIR.mkdir(exist_ok=True)


def _get_metadata_file_path(doc_id: str) -> Path:
    """Get the file path for a document's metadata."""
    return METADATA_DIR / f"{doc_id}.json"


def _load_metadata(doc_id: str) -> Optional[Dict]:
    """Load metadata from file."""
    metadata_file = _get_metadata_file_path(doc_id)
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata for {doc_id}: {e}")
        return None


def _save_metadata(doc_id: str, metadata: Dict) -> None:
    """Save metadata to file."""
    metadata_file = _get_metadata_file_path(doc_id)
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.debug(f"Saved metadata to {metadata_file}")
    except Exception as e:
        logger.error(f"Error saving metadata for {doc_id}: {e}")
        raise


async def save_document_metadata(
    doc_id: str,
    filename: str,
    filepath: str,
    status: str = "processing"
) -> None:
    """
    Save document metadata to file storage.
    
    Args:
        doc_id: Unique document identifier
        filename: Original filename
        filepath: Path where file is stored
        status: Processing status (processing, completed, failed)
    """
    try:
        metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "filepath": filepath,
            "status": status,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "processed_text": None,
            "cleaned_text": None,
            "summary": None
        }
        
        _save_metadata(doc_id, metadata)
        logger.info(f"Saved document metadata: doc_id={doc_id}")
    except Exception as e:
        logger.error(f"Error saving document metadata: {e}")
        raise


async def update_document_status(
    doc_id: str,
    status: str,
    processed_text: Optional[str] = None,
    cleaned_text: Optional[str] = None
) -> None:
    """
    Update document processing status and store processed text.
    
    Args:
        doc_id: Document identifier
        status: New status (processing, completed, failed)
        processed_text: Extracted text from document
        cleaned_text: Cleaned text from document
    """
    try:
        metadata = _load_metadata(doc_id)
        if not metadata:
            logger.warning(f"Metadata not found for doc_id={doc_id}, creating new entry")
            metadata = {
                "doc_id": doc_id,
                "filename": "unknown",
                "filepath": "",
                "status": status,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "processed_text": None,
                "cleaned_text": None,
                "summary": None
            }
        
        metadata["status"] = status
        metadata["updated_at"] = datetime.utcnow().isoformat()
        
        if processed_text is not None:
            metadata["processed_text"] = processed_text
        if cleaned_text is not None:
            metadata["cleaned_text"] = cleaned_text
        
        _save_metadata(doc_id, metadata)
        logger.info(f"Updated document status: doc_id={doc_id}, status={status}")
    except Exception as e:
        logger.error(f"Error updating document status: {e}")
        raise


async def save_document_summary(doc_id: str, summary: str) -> None:
    """
    Save document summary to metadata.
    
    Args:
        doc_id: Document identifier
        summary: Generated summary text
    """
    try:
        metadata = _load_metadata(doc_id)
        if not metadata:
            logger.warning(f"Metadata not found for doc_id={doc_id}")
            return
        
        metadata["summary"] = summary
        metadata["updated_at"] = datetime.utcnow().isoformat()
        
        _save_metadata(doc_id, metadata)
        logger.info(f"Saved summary for document: doc_id={doc_id}")
    except Exception as e:
        logger.error(f"Error saving document summary: {e}")
        raise


async def get_document_metadata(doc_id: str) -> Optional[Dict]:
    """
    Get document metadata by ID.
    
    Args:
        doc_id: Document identifier
    
    Returns:
        Document metadata dictionary or None if not found
    """
    try:
        return _load_metadata(doc_id)
    except Exception as e:
        logger.error(f"Error retrieving document metadata: {e}")
        raise


async def get_all_documents() -> List[Dict]:
    """
    Get all document metadata.
    
    Returns:
        List of document metadata dictionaries
    """
    try:
        documents = []
        if not METADATA_DIR.exists():
            return documents
        
        for metadata_file in METADATA_DIR.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    # Remove large text fields from list view for performance
                    metadata.pop("processed_text", None)
                    metadata.pop("cleaned_text", None)
                    metadata.pop("summary", None)
                    documents.append(metadata)
            except Exception as e:
                logger.warning(f"Error reading metadata file {metadata_file}: {e}")
                continue
        
        # Sort by created_at (newest first)
        documents.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return documents
    except Exception as e:
        logger.error(f"Error retrieving all documents: {e}")
        raise


async def get_document_text(doc_id: str) -> Optional[str]:
    """
    Get cleaned text from document metadata.
    
    Args:
        doc_id: Document identifier
    
    Returns:
        Cleaned text or None if not found
    """
    try:
        metadata = _load_metadata(doc_id)
        if metadata:
            return metadata.get("cleaned_text") or metadata.get("processed_text")
        return None
    except Exception as e:
        logger.error(f"Error retrieving document text: {e}")
        raise


async def delete_document_metadata(doc_id: str) -> bool:
    """
    Delete document metadata file.
    
    Args:
        doc_id: Document identifier
    
    Returns:
        True if deleted, False if not found
    """
    try:
        metadata_file = _get_metadata_file_path(doc_id)
        if metadata_file.exists():
            metadata_file.unlink()
            logger.info(f"Deleted metadata for document: doc_id={doc_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting document metadata: {e}")
        raise
