import json
import os
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.dataset_service import (
    build_column_profile,
    default_audit_columns,
    ensure_data_dirs,
    infer_favorable_value,
    infer_outcome_column,
    load_dataset,
    sanitize_findings,
    UPLOAD_DIR,
)
from services.gemini_service import get_gemini_findings
from services.groq_service import validate_findings_with_claude


router = APIRouter()


@router.post("/")
async def upload_dataset(file: UploadFile = File(...)):
    ensure_data_dirs()
    filename = file.filename or ""

    if not filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only CSV or Excel files are allowed")

    file_id = str(uuid.uuid4())
    safe_name = os.path.basename(filename).replace(" ", "_")
    filepath = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_name}")

    with open(filepath, "wb") as f:
        f.write(await file.read())

    try:
        df = load_dataset(filepath)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    profile = build_column_profile(df)
    columns = df.columns.tolist()
    outcome_column = infer_outcome_column(df)
    favorable_value = infer_favorable_value(df, outcome_column)
    gemini_findings = get_gemini_findings(columns, profile)
    validated_findings = validate_findings_with_claude(columns, profile, gemini_findings)
    final_findings = sanitize_findings(validated_findings, columns, outcome_column, df, favorable_value)
    suggested_sensitive_columns = default_audit_columns(final_findings, df, outcome_column)

    meta = {
        "file_id": file_id,
        "filename": filename,
        "filepath": filepath,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
        "suggested_outcome_column": outcome_column,
        "suggested_favorable_value": favorable_value,
        "suggested_sensitive_columns": suggested_sensitive_columns,
    }

    with open(os.path.join(UPLOAD_DIR, f"{file_id}_findings.json"), "w", encoding="utf-8") as f:
        json.dump(final_findings, f, indent=2)

    with open(os.path.join(UPLOAD_DIR, f"{file_id}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        **meta,
        "findings": final_findings,
    }
