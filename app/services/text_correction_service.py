"""Text correction service using LLM to fix OCR errors.

If the LLM call fails, the uncorrected text is returned
so the document processing pipeline is never blocked.
"""
from app.services.llm_service import generate_llm_response
import logging

logger = logging.getLogger(__name__)


def correct_ocr_text(text: str) -> str:
    prompt = (
        "The following text is extracted using OCR and may contain errors. "
        "Correct only clear errors in spelling, grammar, or OCR misreads. "
        "Do not paraphrase, summarize, omit, or invent information. "
        "Return only the corrected full text.\n\nText:\n" + text
    )
    try:
        return generate_llm_response(prompt)
    except Exception as e:
        logger.warning("OCR correction failed, using raw text: %s", e)
        return text
