"""Document upload and processing routes."""
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
import logging
from datetime import datetime
import time

from services.file_upload_service import save_uploaded_file
from services.document_processor_service import process_document
from services.document_metadata_service import (
    save_document_metadata,
    get_document_metadata,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    doc_id: str
    message: str
    status: str


class DocumentStatusResponse(BaseModel):
    """Response model for document status."""
    doc_id: str
    status: str
    filename: Optional[str] = None
    created_at: Optional[str] = None
    message: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    documents: List[Dict]
    total: int


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=202,
    summary="Upload and process a document",
    description="Upload a PDF document for processing. The document will be processed in the background."
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF document to upload")
) -> DocumentUploadResponse:
    """
    Upload a document and start processing it in the background.
    
    Returns immediately with a document ID that can be used to track processing status.
    """
    try:
        start_time = time.time()
        print(f"start time: {start_time}")
        logger.info(f"start time: {start_time}")
        # Save the uploaded file
        filepath = await save_uploaded_file(file)
        
        # Generate a unique document ID for tracking
        doc_id = str(uuid.uuid4())
        
        # Save document metadata
        await save_document_metadata(
            doc_id=doc_id,
            filename=file.filename or "unknown.pdf",
            filepath=filepath,
            status="processing"
        )
        
        # Start processing in the background
        background_tasks.add_task(process_document, filepath, doc_id)
        
        logger.info(f"Document upload initiated: doc_id={doc_id}, filename={file.filename}")
        end_time = time.time()
        print(f"end time: {end_time}")
        logger.info(f"end time: {end_time}")

        duration_seconds = end_time - start_time
        duration_minutes = duration_seconds / 60

        print(f"Time taken: {duration_seconds:.2f} seconds")
        print(f"Time taken: {duration_minutes:.2f} minutes")
        return DocumentUploadResponse(
            doc_id=doc_id,
            message="Document upload successful. Processing started in background.",
            status="processing"
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload document: {str(e)}"
        )


@router.get(
    "/documents/{doc_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get document processing status",
    description="Check the processing status of a document by its ID."
)
async def get_document_status(doc_id: str) -> DocumentStatusResponse:
    """
    Get the processing status of a document.
    """
    try:
        metadata = await get_document_metadata(doc_id)
        
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {doc_id} not found"
            )
        
        # Convert datetime to string
        created_at = None
        if metadata.get("created_at"):
            created_at_str = metadata["created_at"]
            created_at_dt = datetime.fromisoformat(created_at_str)  # if it’s in ISO 8601 format
            created_at = created_at_dt.isoformat()
        
        return DocumentStatusResponse(
            doc_id=doc_id,
            status=metadata.get("status", "unknown"),
            filename=metadata.get("filename"),
            created_at=created_at,
            message=f"Document status: {metadata.get('status', 'unknown')}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document status: {str(e)}"
        )

