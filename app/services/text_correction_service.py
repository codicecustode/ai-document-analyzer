"""Text correction service using LLM to fix OCR errors."""
from app.services.llm_service import generate_llm_response
import logging

logger = logging.getLogger(__name__)


def correct_ocr_text(text: str) -> str:
    """Correct OCR-extracted text using LLM. Falls back to uncorrected text if LLM fails."""
    prompt = (
        "The following text is extracted using OCR and may contain spelling, grammar, or formatting errors. "
        "Correct only clear errors in spelling, grammar, or OCR misreads. "
        "Do not paraphrase, summarize, omit, or invent information. "
        "Preserve the original structure and wording as much as possible. "
        "Return only the corrected full text, without any explanation.\n\n"
        f"Text:\n{text}"
    )
    try:
        corrected = generate_llm_response(prompt)
        if corrected and len(corrected) > len(text) * 0.3:
            return corrected
        logger.warning("LLM returned too short or empty correction, using original text")
        return text
    except Exception as e:
        logger.warning("OCR text correction failed (%s), using uncorrected text", e)
        return text
