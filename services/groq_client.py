import logging
import os
from typing import Any

logger = logging.getLogger("datalens.groq")

DEFAULT_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
]


def is_groq_configured() -> bool:
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    return bool(key) and not key.lower().startswith("your_")


def _model_candidates() -> list[str]:
    configured = (os.environ.get("GROQ_MODEL") or "").strip()
    names = [configured] if configured else []
    for name in DEFAULT_MODELS:
        if name not in names:
            names.append(name)
    return names


def groq_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.25,
    max_tokens: int = 1200,
) -> tuple[str | None, str | None]:
    """
    Returns (response_text, error_message).
    """
    if not is_groq_configured():
        return None, "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq key."

    try:
        from groq import Groq
    except ImportError as exc:
        return None, f"Groq package not installed: {exc}"

    client = Groq(api_key=os.environ["GROQ_API_KEY"].strip())
    last_error = "No Groq model responded successfully."

    for model_name in _model_candidates():
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                logger.info("Groq response via model %s", model_name)
                return text, None
        except Exception as exc:
            last_error = f"{model_name}: {exc}"
            logger.warning("Groq model %s failed: %s", model_name, exc)

    return None, last_error
