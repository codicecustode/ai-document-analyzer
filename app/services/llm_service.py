"""LLM service for interacting with Google Gemini."""
import logging
from google import genai
from typing import Optional

logger = logging.getLogger(__name__)

# Singleton client instance
_client: Optional[genai.Client] = None


def get_llm_client() -> genai.Client:
    """
    Get or create Gemini LLM client (singleton pattern).
    
    Returns:
        genai.Client: Initialized Gemini client
    """
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def generate_llm_response(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """
    Generate text using Gemini LLM.
    
    Args:
        prompt: The prompt to send to the model
        model: Model name (default: "gemini-2.5-flash")
    
    Returns:
        str: Generated text response
    
    Raises:
        RuntimeError: If generation fails
    """
    try:
        client = get_llm_client()
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Error generating text with {model}: {e}")
        raise RuntimeError(f"LLM generation failed: {e}") from e

