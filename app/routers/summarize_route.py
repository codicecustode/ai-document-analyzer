"""Document summarization routes with model selection."""
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from app.services.document_extraction_service import extract_text_with_ocr
from app.services.text_cleaning_service import clean_ocr_text
from app.services.summarizer_service import summarize_text
from app.services.document_metadata_service import get_document_text, get_document_metadata, save_document_summary

logger = logging.getLogger(__name__)
router = APIRouter()


class SummarizeRequest(BaseModel):
    text: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


class SummarizeResponse(BaseModel):
    summary: str
    provider: str
    model: str
    original_length: Optional[int] = None
    summary_length: Optional[int] = None


@router.post("/summarize/{doc_id}", response_model=SummarizeResponse, status_code=200)
async def summarize_uploaded_document(doc_id: str, provider: Optional[str] = None, model: Optional[str] = None) -> SummarizeResponse:
    try:
        metadata = await get_document_metadata(doc_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
        if metadata.get("status") != "completed":
            raise HTTPException(status_code=400, detail=f"Document still processing: {metadata.get('status')}")

        text = await get_document_text(doc_id)
        if not text:
            raise HTTPException(status_code=400, detail="Document text not available.")

        summary, used_provider, used_model = summarize_text(text, provider=provider, model=model)
        await save_document_summary(doc_id, summary)

        return SummarizeResponse(summary=summary, provider=used_provider, model=used_model, original_length=len(text), summary_length=len(summary))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Summarize error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize", response_model=SummarizeResponse, status_code=200)
async def summarize_new_document(file: UploadFile = File(...), provider: Optional[str] = None, model: Optional[str] = None) -> SummarizeResponse:
    try:
        import os, tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        try:
            extracted = extract_text_with_ocr(tmp_path)
            cleaned = clean_ocr_text(extracted)
            summary, used_provider, used_model = summarize_text(cleaned, provider=provider, model=model)
            return SummarizeResponse(summary=summary, provider=used_provider, model=used_model, original_length=len(extracted), summary_length=len(summary))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        logger.error("Summarize upload error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize/text", response_model=SummarizeResponse, status_code=200)
async def summarize_plain_text(request: SummarizeRequest) -> SummarizeResponse:
    if not request.text:
        raise HTTPException(status_code=400, detail="Text field is required")
    try:
        summary, used_provider, used_model = summarize_text(request.text, provider=request.provider, model=request.model)
        return SummarizeResponse(summary=summary, provider=used_provider, model=used_model, original_length=len(request.text), summary_length=len(summary))
    except Exception as e:
        logger.error("Summarize text error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
