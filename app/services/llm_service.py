"""LLM service supporting Google Gemini, Groq and NVIDIA.

All three providers accept the OpenAI chat-completions format, so one client
works for all of them. The provider is chosen by setting LLM_PROVIDER in .env.
"""
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-20b",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        "default_model": "deepseek-ai/deepseek-v4.1-flash",
    },
}

_provider_id = os.getenv("LLM_PROVIDER", "gemini").lower()
if _provider_id not in PROVIDERS:
    raise ValueError(f"Unknown LLM_PROVIDER '{_provider_id}'. Choose: {', '.join(PROVIDERS)}")

_config = PROVIDERS[_provider_id]
_api_key = os.getenv(_config["key_env"], "")
if not _api_key:
    raise ValueError(f"{_config['key_env']} is not set. Add it to .env and restart.")

_client = OpenAI(base_url=_config["base_url"], api_key=_api_key, timeout=120)
_default_model = os.getenv("LLM_MODEL", _config["default_model"])

logger.info("LLM provider: %s, model: %s", _provider_id, _default_model)


def generate_llm_response(prompt: str, model: str | None = None) -> str:
    """
    Generate text using the configured LLM provider.

    Args:
        prompt: The prompt to send to the model
        model: Model name override (defaults to LLM_MODEL or the provider's default)

    Returns:
        str: Generated text response

    Raises:
        RuntimeError: If generation fails
    """
    model = model or _default_model
    try:
        response = _client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Error generating text with %s/%s: %s", _provider_id, model, e)
        raise RuntimeError(f"LLM generation failed: {e}") from e
