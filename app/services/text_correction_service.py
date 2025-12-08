"""Text correction service using LLM to fix OCR errors."""
from app.services.llm_service import generate_llm_response
import logging

logger = logging.getLogger(__name__)


def correct_ocr_text(text: str) -> str:
    """
    Correct OCR-extracted text using LLM.
    
    Only fixes clear errors in spelling, grammar, or OCR misreads.
    Does not paraphrase, summarize, or invent information.
    
    Args:
        text: Text extracted via OCR that may contain errors
    
    Returns:
        str: Corrected text with errors fixed
    
    Raises:
        RuntimeError: If correction fails
    """
    prompt = (
        "The following text is extracted using OCR and may contain spelling, grammar, or formatting errors. "
        "Correct only clear errors in spelling, grammar, or OCR misreads. "
        "Do not paraphrase, summarize, omit, or invent information. "
        "Preserve the original structure and wording as much as possible. "
        "Return only the corrected full text, without any explanation, introductory, or summary sentences.\n\n"
        f"Text:\n{text}"
    )
    
    try:
        corrected_text = generate_llm_response(prompt)
        return corrected_text
    except Exception as e:
        logger.error(f"Error correcting OCR text: {e}")
        raise RuntimeError(f"Text correction failed: {e}") from e

