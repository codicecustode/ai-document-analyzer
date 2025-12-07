"""Document summarization routes."""
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from services.document_extraction_service import extract_text_with_ocr
from services.text_cleaning_service import clean_ocr_text
from services.summarizer_service import summarize_with_gemini
from services.document_metadata_service import (
    get_document_text,
    get_document_metadata,
    save_document_summary
)

logger = logging.getLogger(__name__)
router = APIRouter()


class SummarizeRequest(BaseModel):
    """Request model for summarization."""
    text: Optional[str] = None
    doc_id: Optional[str] = None


class SummarizeResponse(BaseModel):
    """Response model for summarization."""
    summary: str
    original_length: Optional[int] = None
    summary_length: Optional[int] = None


@router.post(
    "/summarize/{doc_id}",
    response_model=SummarizeResponse,
    status_code=200,
    summary="Summarize an uploaded document",
    description="Summarize an already uploaded document by its ID. Uses the processed text from the document."
)
async def summarize_uploaded_document(doc_id: str) -> SummarizeResponse:
    """
    Summarize an already uploaded and processed document.
    
    This endpoint uses the cleaned text from the document that was already processed.
    No need to re-upload the file.
    """
    try:
        # Check if document exists
        metadata = await get_document_metadata(doc_id)
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {doc_id} not found. Please upload the document first."
            )
        
        # Check if document is processed
        if metadata.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Document is still processing. Current status: {metadata.get('status')}"
            )
        
        # Get cleaned text from metadata
        text = await get_document_text(doc_id)
        if not text:
            raise HTTPException(
                status_code=400,
                detail="Document text not available. Document may not be fully processed."
            )
        
        # Check if summary already exists
        if metadata.get("summary"):
            logger.info(f"Returning cached summary for document: {doc_id}")
            return SummarizeResponse(
                summary=metadata["summary"],
                original_length=len(text),
                summary_length=len(metadata["summary"])
            )
        
        # Generate summary
        logger.info(f"Generating summary for document: {doc_id}")
        summary = summarize_with_gemini(text)
        
        # Save summary to metadata
        await save_document_summary(doc_id, summary)
        
        return SummarizeResponse(
            summary=summary,
            original_length=len(text),
            summary_length=len(summary)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to summarize document: {str(e)}"
        )


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=200,
    summary="Summarize a new document (upload)",
    description="Upload a PDF document and get a summarized version. For one-time summarization without storing."
)
async def summarize_new_document(
    file: UploadFile = File(..., description="PDF document to summarize")
) -> SummarizeResponse:
    """
    Summarize a PDF document.
    
    This endpoint:
    1. Extracts text from PDF using OCR
    2. Cleans the extracted text
    3. Generates a summary using AI
    """
    try:
        import os
        import tempfile
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Step 1: Extract text with OCR
            logger.info(f"Extracting text from PDF: {file.filename}")
            extracted_text = extract_text_with_ocr(tmp_file_path)
            original_length = len(extracted_text)
            
            # Step 2: Clean OCR text
            logger.debug("Cleaning extracted text...")
            cleaned_text = clean_ocr_text(extracted_text)
            
            # Step 3: Generate summary
            logger.info("Generating summary with AI...")
            summary = summarize_with_gemini(cleaned_text)
            summary_length = len(summary)
            
            logger.info(f"Summary generated: {summary_length} characters from {original_length} characters")
            
            return SummarizeResponse(
                summary=summary,
                original_length=original_length,
                summary_length=summary_length
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        logger.error(f"Error summarizing document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to summarize document: {str(e)}"
        )


@router.post(
    "/summarize/text",
    response_model=SummarizeResponse,
    status_code=200,
    summary="Summarize plain text",
    description="Summarize plain text using AI. For one-time summarization without storing."
)
async def summarize_text(request: SummarizeRequest) -> SummarizeResponse:
    """
    Summarize plain text.
    
    This endpoint takes text directly and generates a summary.
    """
    try:
        if not request.text:
            raise HTTPException(
                status_code=400,
                detail="Text field is required"
            )
        
        original_length = len(request.text)
        
        logger.info(f"Summarizing text: {original_length} characters")
        summary = summarize_with_gemini(request.text)
        summary_length = len(summary)
        
        return SummarizeResponse(
            summary=summary,
            original_length=original_length,
            summary_length=summary_length
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing text: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to summarize text: {str(e)}"
        )
