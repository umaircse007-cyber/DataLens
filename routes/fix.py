import json
import os

from fastapi import APIRouter, Body, HTTPException

from services.dataset_service import UPLOAD_DIR, load_dataset
from services.fix_service import apply_multi_column_fix
from services.metrics_service import calculate_fairness_metrics


router = APIRouter()


@router.post("/")
async def apply_fix(
    file_id: str = Body(...),
    filepath: str = Body(...),
    sensitive_columns: list[str] = Body(...),
    outcome_column: str = Body(...),
    favorable_value: str = Body(...),
):
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = load_dataset(filepath)
    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    if outcome_column not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid outcome column")

    fixed_df = apply_multi_column_fix(df, sensitive_columns, outcome_column, favorable_value)
    fixed_filepath = os.path.join(UPLOAD_DIR, f"{file_id}_fixed.csv")
    fixed_df.to_csv(fixed_filepath, index=False)

    original_metrics = calculate_fairness_metrics(filepath, sensitive_columns, outcome_column, favorable_value)
    fixed_metrics = calculate_fairness_metrics(fixed_filepath, sensitive_columns, outcome_column, favorable_value)

    with open(os.path.join(UPLOAD_DIR, f"{file_id}_fixed_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(fixed_metrics, f, indent=2)

    return {
        "original_metrics": original_metrics,
        "fixed_metrics": fixed_metrics,
        "fixed_filepath": fixed_filepath,
        "download_url": f"/download/fixed/{file_id}",
    }
