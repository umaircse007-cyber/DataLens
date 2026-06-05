import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from typing import List

import pandas as pd
from fastapi import APIRouter, Body, Cookie, File, HTTPException, Response, UploadFile
from pydantic import BaseModel

from services.ai_assistant_service import ai_provider_status, answer_audit_question
from services.analysis_service import run_full_analysis, run_multi_table_analysis
from services.dataset_service import DATA_DIR, UPLOAD_DIR, ensure_data_dirs
from services.dictionary_cache import get_result, store_result, update_result
from services.security_service import decrypt_to_memory, save_encrypted
from routes.export import export_result


class AskPayload(BaseModel):
    question: str
    session_id: str | None = None
    result: dict | None = None


router = APIRouter(prefix="/dictionary", tags=["Dictionary"])

HISTORY_COOKIE_NAME = "datalens_history_session"
HISTORY_COOKIE_SECONDS = 3600


def _load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    buffer = BytesIO(file_bytes)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    if name.endswith(".csv"):
        return pd.read_csv(buffer)
    raise ValueError("Only CSV or Excel files are allowed")


def _history_db_path() -> str:
    database_url = os.environ.get("DATABASE_URL")
    if database_url and database_url.startswith("sqlite:///"):
        return database_url.replace("sqlite:///", "", 1)
    return str(DATA_DIR / "datalens.db")


def _ensure_history_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS dictionary_history (
            session_id TEXT PRIMARY KEY,
            owner_session_id TEXT,
            filename TEXT,
            timestamp TEXT,
            row_count INTEGER,
            column_count INTEGER,
            readiness_score INTEGER,
            flagged_column_count INTEGER
        )
        """
    )
    columns = connection.execute("PRAGMA table_info(dictionary_history)").fetchall()
    existing_column_names = {row[1] for row in columns}
    if "owner_session_id" not in existing_column_names:
        connection.execute("ALTER TABLE dictionary_history ADD COLUMN owner_session_id TEXT")


def _cookie_is_secure() -> bool:
    return os.environ.get("ENABLE_HTTPS_REDIRECT", "").strip().lower() in {"1", "true", "yes"}


def _history_owner_id(existing_cookie: str | None, response: Response) -> str:
    owner_session_id = (existing_cookie or "").strip() or str(uuid.uuid4())
    response.set_cookie(
        key=HISTORY_COOKIE_NAME,
        value=owner_session_id,
        max_age=HISTORY_COOKIE_SECONDS,
        httponly=True,
        secure=_cookie_is_secure(),
        samesite="lax",
    )
    return owner_session_id


def _save_dictionary_history(result: dict, owner_session_id: str) -> None:
    metadata = result["metadata"]
    readiness = result["readiness"]
    db_path = _history_db_path()
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        _ensure_history_schema(connection)
        cutoff = (datetime.utcnow() - timedelta(seconds=HISTORY_COOKIE_SECONDS)).isoformat()
        connection.execute("DELETE FROM dictionary_history WHERE timestamp < ?", (cutoff,))
        connection.execute(
            """
            INSERT OR REPLACE INTO dictionary_history (
                session_id, owner_session_id, filename, timestamp, row_count, column_count,
                readiness_score, flagged_column_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata["session_id"],
                owner_session_id,
                metadata["filename"],
                metadata["timestamp"],
                metadata["row_count"],
                metadata["column_count"],
                readiness["score"],
                metadata["flagged_column_count"],
            ),
        )
        connection.commit()


@router.post("/analyse")
async def analyse_dictionary(
    response: Response,
    file: UploadFile = File(...),
    datalens_history_session: str | None = Cookie(default=None, alias=HISTORY_COOKIE_NAME),
):
    ensure_data_dirs()
    owner_session_id = _history_owner_id(datalens_history_session, response)
    filename = file.filename or ""
    if not filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only CSV or Excel files are allowed")

    try:
        encrypted_session_id, encrypted_path = save_encrypted(await file.read(), UPLOAD_DIR)
        file_bytes = decrypt_to_memory(encrypted_path)
        df = _load_dataframe(file_bytes, filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset securely: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    result = await run_full_analysis(df, filename, encrypted_session_id)

    store_result(encrypted_session_id, result)
    _save_dictionary_history(result, owner_session_id)
    return result


@router.post("/analyse-multi")
async def analyse_dictionary_multi(
    response: Response,
    files: List[UploadFile] = File(...),
    datalens_history_session: str | None = Cookie(default=None, alias=HISTORY_COOKIE_NAME),
):
    ensure_data_dirs()
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    owner_session_id = _history_owner_id(datalens_history_session, response)
    tables: list[tuple[str, pd.DataFrame]] = []
    encrypted_session_id = None
    encrypted_path = None

    for index, upload in enumerate(files):
        filename = upload.filename or ""
        if not filename.lower().endswith((".csv", ".xlsx", ".xls")):
            raise HTTPException(status_code=400, detail="Only CSV or Excel files are allowed")
        raw_bytes = await upload.read()
        if index == 0:
            encrypted_session_id, encrypted_path = save_encrypted(raw_bytes, UPLOAD_DIR)
            file_bytes = raw_bytes
        else:
            file_bytes = raw_bytes
        frame = _load_dataframe(file_bytes, filename)
        if frame.empty:
            raise HTTPException(status_code=400, detail=f"Dataset {filename} is empty")
        tables.append((filename, frame))

    primary_filename = tables[0][0]
    result = await run_multi_table_analysis(tables, encrypted_session_id, primary_filename)
    store_result(encrypted_session_id, result)
    _save_dictionary_history(result, owner_session_id)
    return result


@router.get("/ai-status")
async def dictionary_ai_status():
    return ai_provider_status()


@router.post("/ask")
async def ask_dictionary_ai(payload: AskPayload):
    question = str(payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    session_id = str(payload.session_id or "").strip()
    result = get_result(session_id) if session_id else None
    if not result and isinstance(payload.result, dict):
        result = payload.result
    if not isinstance(result, dict):
        raise HTTPException(status_code=404, detail="No audit result found for this question")

    if not result.get("chat_index"):
        from services.chat_index_service import build_chat_index
        result["chat_index"] = build_chat_index(result)

    return answer_audit_question(question, result)


@router.get("/export/{session_id}/{format}")
async def export_dictionary_result(session_id: str, format: str):
    result = get_result(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Session result not found")
    return export_result(session_id, format, result)


@router.post("/export/{session_id}/{format}")
async def export_edited_dictionary_result(session_id: str, format: str, result: dict = Body(...)):
    update_result(session_id, result)
    return export_result(session_id, format, result)


@router.get("/history/list")
async def list_dictionary_history(
    datalens_history_session: str | None = Cookie(default=None, alias=HISTORY_COOKIE_NAME),
):
    if not datalens_history_session:
        return []

    owner_session_id = datalens_history_session.strip()
    if not owner_session_id:
        return []

    db_path = _history_db_path()
    if not os.path.exists(db_path):
        return []

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_history_schema(connection)
        cutoff = (datetime.utcnow() - timedelta(seconds=HISTORY_COOKIE_SECONDS)).isoformat()
        connection.execute("DELETE FROM dictionary_history WHERE timestamp < ?", (cutoff,))
        rows = connection.execute(
            """
            SELECT session_id, filename, timestamp, row_count, column_count,
                   readiness_score, flagged_column_count
            FROM dictionary_history
            WHERE owner_session_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 100
            """,
            (owner_session_id, cutoff),
        ).fetchall()
        connection.commit()
    return [dict(row) for row in rows]
