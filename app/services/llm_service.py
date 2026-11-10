"""LLM providers the document analyzer can use.

Every provider accepts the OpenAI chat-completions format, so one client
works for all of them. The frontend picks the provider and model per request.
A provider becomes available as soon as its API key is set in the environment.
"""
import logging
import os
import re

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

PROVIDERS = {
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "max_input_chars": 400_000,
        "models": {
            "gemini-2.5-flash": "Gemini 2.5 Flash",
            "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
        },
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "max_input_chars": 20_000,
        "models": {
            "openai/gpt-oss-20b": "GPT-OSS 20B",
            "openai/gpt-oss-120b": "GPT-OSS 120B",
            "qwen/qwen3.8-27b": "Qwen3.8 27B",
        },
    },
    "nvidia": {
        "label": "NVIDIA",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        "max_input_chars": 100_000,
        "models": {
            "deepseek-ai/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
            "nvidia/nemotron-3.5-lightning-30b-a3b": "Nemotron 3.5 Lightning 30B",
            "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra 550B",
        },
    },
}

DEFAULT_CHOICE = ("gemini", "gemini-2.5-flash")

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Reuse one client per provider
_clients: dict[str, OpenAI] = {}


def _api_key(provider: str) -> str:
    return os.getenv(PROVIDERS[provider]["key_env"], "").strip()


def _get_client(provider: str) -> OpenAI:
    if provider not in _clients:
        config = PROVIDERS[provider]
        key = _api_key(provider)
        if not key:
            raise RuntimeError(f"{config['key_env']} is not set.")
        _clients[provider] = OpenAI(base_url=config["base_url"], api_key=key, timeout=120, max_retries=1)
    return _clients[provider]


def list_models() -> list[dict]:
    """Providers and their models for the frontend."""
    return [
        {
            "id": pid,
            "label": p["label"],
            "available": bool(_api_key(pid)),
            "models": [{"id": mid, "label": ml} for mid, ml in p["models"].items()],
        }
        for pid, p in PROVIDERS.items()
    ]


def default_choice() -> tuple[str, str] | None:
    if _api_key(DEFAULT_CHOICE[0]):
        return DEFAULT_CHOICE
    for pid, p in PROVIDERS.items():
        if _api_key(pid):
            return pid, next(iter(p["models"]))
    return None


def resolve_provider_model(provider: str | None, model: str | None) -> tuple[str, str]:
    """Resolve a frontend choice to a valid (provider, model) pair, or raise."""
    if not provider or not model:
        choice = default_choice()
        if choice is None:
            raise RuntimeError("No API keys are set. Add GEMINI_API_KEY, GROQ_API_KEY or NVIDIA_API_KEY.")
        return choice
    if provider not in PROVIDERS:
        raise RuntimeError(f"Unknown provider: {provider}")
    if model not in PROVIDERS[provider]["models"]:
        raise RuntimeError(f"{PROVIDERS[provider]['label']} has no model called {model}.")
    if not _api_key(provider):
        raise RuntimeError(f"{PROVIDERS[provider]['key_env']} is not set.")
    return provider, model


def generate_llm_response(prompt: str, model: str | None = None, provider: str | None = None) -> str:
    """Generate text using a single prompt (no conversation history)."""
    provider, model = resolve_provider_model(provider, model)
    client = _get_client(provider)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = THINK_BLOCK.sub("", response.choices[0].message.content or "").strip()
        return text
    except Exception as e:
        logger.error("LLM error (%s/%s): %s", provider, model, e)
        raise RuntimeError(f"LLM generation failed: {e}") from e


def chat_llm_response(messages: list[dict], provider: str | None = None, model: str | None = None) -> str:
    """Generate a response from a full conversation (with history)."""
    provider, model = resolve_provider_model(provider, model)
    client = _get_client(provider)
    try:
        response = client.chat.completions.create(model=model, messages=messages)
        text = THINK_BLOCK.sub("", response.choices[0].message.content or "").strip()
        return text
    except Exception as e:
        logger.error("LLM chat error (%s/%s): %s", provider, model, e)
        raise RuntimeError(f"LLM generation failed: {e}") from e
