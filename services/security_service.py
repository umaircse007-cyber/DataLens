import logging
import os
import threading
import time
import uuid
from typing import Any

import pandas as pd
from cryptography.fernet import Fernet

logger = logging.getLogger("datalens.security")

PLACEHOLDER_MARKERS = (
    "generate_with",
    "your_",
    "changeme",
    "replace_me",
    "paste_",
)


def _looks_like_placeholder(key: str) -> bool:
    lower = key.lower().strip()
    if not lower:
        return True
    return any(marker in lower for marker in PLACEHOLDER_MARKERS)


def _is_valid_fernet_key(key: str) -> bool:
    if _looks_like_placeholder(key):
        return False
    try:
        Fernet(key.encode("utf-8"))
        return True
    except Exception:
        return False


def ensure_file_encryption_key() -> str:
    """
    Return a valid Fernet key. If .env lacks a valid key, generate one and persist it to .env when possible.
    Falls back to an in-memory key if writing fails.
    """
    from services.groq_keys import ensure_env_loaded

    ensure_env_loaded()
    key = (os.environ.get("FILE_ENCRYPTION_KEY") or "").strip()
    if _is_valid_fernet_key(key):
        return key

    # Generate a new Fernet key and persist it to .env if possible
    key = Fernet.generate_key().decode()
    os.environ["FILE_ENCRYPTION_KEY"] = key

    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        env_path = os.path.join(base_dir, ".env")

        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                text = f.read()
            if "FILE_ENCRYPTION_KEY=" in text:
                lines = text.splitlines()
                replaced = False
                for i, line in enumerate(lines):
                    if line.strip().startswith("FILE_ENCRYPTION_KEY="):
                        lines[i] = f"FILE_ENCRYPTION_KEY={key}"
                        replaced = True
                        break
                if not replaced:
                    lines.append(f"FILE_ENCRYPTION_KEY={key}")
                text = "\n".join(lines) + "\n"
            else:
                if not text.endswith("\n") and text:
                    text += "\n"
                text += f"FILE_ENCRYPTION_KEY={key}\n"
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.warning("FILE_ENCRYPTION_KEY was missing or invalid. Generated and saved to .env.")
        else:
            # Create a new .env file with the key
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(f"FILE_ENCRYPTION_KEY={key}\n")
            logger.warning("FILE_ENCRYPTION_KEY generated and written to new .env file.")
    except Exception as exc:
        logger.warning(
            "FILE_ENCRYPTION_KEY was missing or invalid. Using a temporary in-memory key for this server session. Failed to persist to .env: %s",
            exc,
        )

    return key


def _fernet() -> Fernet:
    return Fernet(ensure_file_encryption_key().encode("utf-8"))


def save_encrypted(file_bytes: bytes, upload_dir: str) -> tuple[str, str]:
    session_id = str(uuid.uuid4())
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, f"{session_id}.enc")

    encrypted = _fernet().encrypt(file_bytes)
    with open(path, "wb") as encrypted_file:
        encrypted_file.write(encrypted)

    schedule_deletion(path, seconds=3600)
    return session_id, path


def decrypt_to_memory(path: str) -> bytes:
    with open(path, "rb") as encrypted_file:
        encrypted = encrypted_file.read()
    return _fernet().decrypt(encrypted)


def schedule_deletion(path: str, seconds: int = 3600) -> None:
    def delete_later() -> None:
        time.sleep(seconds)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    thread = threading.Thread(target=delete_later, daemon=True)
    thread.start()


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def build_safe_payload(df: pd.DataFrame, col: str) -> dict:
    series = df[col]
    non_null = series.dropna()
    sample_size = min(5, len(non_null))
    samples = non_null.sample(n=sample_size, random_state=42).tolist() if sample_size else []

    return {
        "column_name": str(col),
        "dtype": str(series.dtype),
        "sample_values": [_json_safe(value) for value in samples],
        "null_pct": round(float(series.isna().mean() * 100), 1),
        "unique_count": int(non_null.nunique()),
    }
