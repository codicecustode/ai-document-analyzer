from dotenv import load_dotenv
from services.llm_service import generate_llm_response
import logging

load_dotenv()  # Load variables from .env file into environment
logger = logging.getLogger(__name__)

def summarize_with_gemini(text: str) -> str:
    # Construct a clear prompt for Gemini
    prompt = (
      "The following text is extracted via OCR and may contain spelling or grammar errors."
      "Carefully correct obvious errors but do not add or invent new information."
      "Summarize all key points concisely, using bullet points or short sections."
      "Omit any introductory text, explanations, or 'Here is a summary...' commentary."
      "Retain the original meaning and structure where possible, but improve readability and clarity."
      "If the document has sections (offers, terms, names, dates, policies, conditions, etc.), separate them with headings or clear bullets."
      "Only output the cleaned, summarized document content."
      "\n\nText:\n" + text
    )


    try:
      return generate_llm_response(prompt, model="gemini-2.5-flash")
    except Exception as e:
      logger.error(f"Error summarizing text with Gemini: {e}")
      raise
