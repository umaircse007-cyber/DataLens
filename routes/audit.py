import json
import os

from fastapi import APIRouter, Body, HTTPException

from services.dataset_service import UPLOAD_DIR, filter_core_audit_columns, load_dataset
from services.metrics_service import calculate_fairness_metrics


router = APIRouter()


@router.post("/")
async def run_audit(
    file_id: str = Body(...),
    filepath: str = Body(...),
    sensitive_columns: list[str] = Body(default=[]),
    outcome_column: str = Body(...),
    favorable_value: str = Body(...),
):
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = load_dataset(filepath)
    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    findings_path = os.path.join(UPLOAD_DIR, f"{file_id}_findings.json")
    findings = []
    if os.path.exists(findings_path):
        with open(findings_path, "r", encoding="utf-8") as f:
            findings = json.load(f)

    filtered_sensitive_columns = filter_core_audit_columns(
        sensitive_columns,
        findings,
        df,
        outcome_column,
    )

    metrics = calculate_fairness_metrics(
        filepath,
        filtered_sensitive_columns,
        outcome_column,
        favorable_value,
    )

    with open(os.path.join(UPLOAD_DIR, f"{file_id}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "mode": "full_audit",
        "metrics": metrics,
        "selected_sensitive_columns": filtered_sensitive_columns,
    }
