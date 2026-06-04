import json
import os

from fastapi import APIRouter, Body

from services.dataset_service import UPLOAD_DIR
from services.eu_mapper_service import map_eu_clauses
from services.report_service import safe_load_json


router = APIRouter()


@router.post("/")
async def eu_map(file_id: str = Body(..., embed=True)):
    metrics = safe_load_json(os.path.join(UPLOAD_DIR, f"{file_id}_metrics.json"), {})
    cf_data = safe_load_json(os.path.join(UPLOAD_DIR, f"{file_id}_cf.json"), {})
    findings = safe_load_json(os.path.join(UPLOAD_DIR, f"{file_id}_findings.json"), [])

    clauses = map_eu_clauses(metrics, float(cf_data.get("flip_rate", 0)), findings)

    with open(os.path.join(UPLOAD_DIR, f"{file_id}_eu.json"), "w", encoding="utf-8") as f:
        json.dump(clauses, f, indent=2)

    return {"triggered_clauses": clauses}
