"""Document summarization service."""
import logging
from app.services.llm_service import generate_llm_response, resolve_provider_model

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = (
    "The following text is extracted via OCR and may contain spelling or grammar errors. "
    "Carefully correct obvious errors but do not add or invent new information. "
    "Summarize all key points concisely, using bullet points or short sections. "
    "Omit any introductory text, explanations, or 'Here is a summary...' commentary. "
    "Retain the original meaning and structure where possible, but improve readability and clarity. "
    "If the document has sections (offers, terms, names, dates, policies, conditions, etc.), "
    "separate them with headings or clear bullets. "
    "Only output the cleaned, summarized document content.\n\nText:\n"
)


def summarize_text(text: str, provider: str | None = None, model: str | None = None) -> tuple[str, str, str]:
    """Summarize text. Returns (summary, provider_used, model_used)."""
    provider, model = resolve_provider_model(provider, model)
    summary = generate_llm_response(SUMMARIZE_PROMPT + text, model=model, provider=provider)
    return summary, provider, model
