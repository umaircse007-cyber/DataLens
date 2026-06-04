import os
from pathlib import Path

from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parents[1]


def ensure_env_loaded() -> None:
    """Load .env.example defaults, then .env overrides (either file may hold API keys)."""
    load_dotenv(_BASE_DIR / ".env.example", override=False)
    load_dotenv(_BASE_DIR / ".env", override=True)
