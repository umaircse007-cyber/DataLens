import os
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import pointbiserialr


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = str(DATA_DIR / "uploads")
REPORT_DIR = str(DATA_DIR / "reports")

PROTECTED_HINTS = {
    "gender",
    "sex",
    "age",
    "age_group",
    "race",
    "ethnicity",
    "religion",
    "disability",
    "nationality",
    "citizenship",
    "marital",
    "pregnancy",
}

IDENTIFIER_HINTS = {
    "id",
    "identifier",
    "uuid",
    "applicant_id",
    "candidate_id",
    "employee_id",
    "user_id",
    "record_id",
}

MERIT_BASED_HINTS = {
    "education",
    "degree",
    "experience",
    "salary",
    "compensation",
    "skill",
    "skills",
    "score",
    "merit",
    "qualification",
    "certification",
    "performance",
    "tenure",
    "interview",
    "assessment",
}

STRONG_PROXY_HINTS = {
    "zip",
    "postal",
    "postcode",
    "address",
    "city",
    "state",
    "name",
    "surname",
    "nationality",
    "citizenship",
}


def ensure_data_dirs() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def load_dataset(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()

    if ext in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(filepath)
        except ImportError as exc:
            raise ImportError(
                "Excel upload requires openpyxl. Run: .\\.venv\\Scripts\\python.exe -m pip install openpyxl"
            ) from exc

    return pd.read_csv(filepath)


def json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def coerce_favorable_value(series: pd.Series, favorable_value):
    if pd.api.types.is_numeric_dtype(series):
        try:
            if "." in str(favorable_value):
                return float(favorable_value)
            return int(favorable_value)
        except Exception:
            return favorable_value
    return str(favorable_value)


def build_outcome_binary(series: pd.Series, favorable_value) -> pd.Series:
    favorable = coerce_favorable_value(series, favorable_value)
    if pd.api.types.is_numeric_dtype(series):
        return (series == favorable).astype(int)
    return (series.astype(str).str.strip().str.lower() == str(favorable).strip().lower()).astype(int)


def normalize_column_name(column: str) -> str:
    return str(column).strip().lower().replace(" ", "_")


def is_identifier_column(column: str) -> bool:
    key = normalize_column_name(column)
    return any(hint == key or hint in key for hint in IDENTIFIER_HINTS)


def is_merit_based_column(column: str) -> bool:
    key = normalize_column_name(column)
    return any(hint in key for hint in MERIT_BASED_HINTS)


def is_protected_column(column: str) -> bool:
    key = normalize_column_name(column)
    return any(hint == key or key.endswith(f"_{hint}") or hint in key for hint in PROTECTED_HINTS)


def is_strong_proxy_column(column: str) -> bool:
    key = normalize_column_name(column)
    return any(hint in key for hint in STRONG_PROXY_HINTS)


def is_continuous_numeric(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False

    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_count = non_null.nunique()
    if unique_count >= 15:
        return True

    return unique_count >= max(8, int(len(non_null) * 0.1))


def encode_series_for_correlation(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    values = series.astype(str).replace({"nan": None})
    codes, _ = pd.factorize(values, sort=True)
    encoded = pd.Series(codes, index=series.index, dtype="float64")
    encoded[encoded < 0] = pd.NA
    return encoded


def correlation_gate(df: pd.DataFrame, column: str, outcome_column: str, favorable_value) -> dict:
    if column not in df.columns or outcome_column not in df.columns:
        return {"r": 0.0, "p": 1.0, "passes": False, "n": 0}

    encoded = encode_series_for_correlation(df[column])
    outcome_binary = build_outcome_binary(df[outcome_column], favorable_value)
    pair = pd.DataFrame({"x": encoded, "y": outcome_binary}).dropna()

    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return {"r": 0.0, "p": 1.0, "passes": False, "n": int(len(pair))}

    try:
        r, p = pointbiserialr(pair["x"], pair["y"])
    except Exception:
        return {"r": 0.0, "p": 1.0, "passes": False, "n": int(len(pair))}

    if pd.isna(r) or pd.isna(p):
        return {"r": 0.0, "p": 1.0, "passes": False, "n": int(len(pair))}

    return {
        "r": float(r),
        "p": float(p),
        "passes": abs(float(r)) >= 0.10 and float(p) <= 0.05,
        "n": int(len(pair)),
    }


def sanitize_findings(findings: list[dict], columns: list[str], outcome_column: str, df: pd.DataFrame, favorable_value) -> list[dict]:
    valid_columns = {str(col): col for col in columns}
    seen = set()
    cleaned = []

    for finding in findings or []:
        raw_column = str(finding.get("column", "")).strip()
        if raw_column not in valid_columns:
            continue
        column = valid_columns[raw_column]
        if column == outcome_column:
            continue
        if is_identifier_column(column):
            continue

        finding_type = str(finding.get("type", "proxy")).lower()
        if finding_type == "sensitive" and not is_protected_column(column):
            if is_merit_based_column(column):
                continue
            finding_type = "proxy" if is_strong_proxy_column(column) else ""
        elif finding_type == "proxy" and is_merit_based_column(column) and not is_strong_proxy_column(column):
            continue

        if not finding_type:
            continue

        stats = correlation_gate(df, column, outcome_column, favorable_value)
        evidence_status = "Flagged" if stats["passes"] else "Clean"
        confidence = str(finding.get("confidence", "Medium"))
        if evidence_status == "Clean" and confidence.lower() == "high":
            confidence = "Medium"

        key = (column, finding_type)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({
            "column": column,
            "type": finding_type,
            "reason": str(finding.get("reason", "Potential fairness-relevant column.")),
            "source": str(finding.get("source", "DataLens")),
            "confidence": confidence,
            "recommended": finding_type == "sensitive",
            "evidence_status": evidence_status,
            "correlation_passes": bool(stats["passes"]),
            "correlation_r": round(stats["r"], 4),
            "correlation_p": round(stats["p"], 4),
            "sample_size": stats["n"],
        })

    return cleaned


def default_audit_columns(findings: list[dict], df: pd.DataFrame, outcome_column: str) -> list[str]:
    selected = []

    for finding in findings:
        column = finding.get("column")
        if not column or column == outcome_column or column not in df.columns:
            continue
        if not finding.get("recommended"):
            continue
        if is_identifier_column(column):
            continue
        if finding.get("type") == "sensitive":
            selected.append(column)

    filtered = []
    for column in selected:
        if column not in df.columns:
            continue
        filtered.append(column)

    return filtered


def filter_core_audit_columns(selected_columns: list[str], findings: list[dict], df: pd.DataFrame, outcome_column: str) -> list[str]:
    finding_by_column = {item.get("column"): item for item in findings or []}
    filtered = []

    for column in selected_columns or []:
        if column not in df.columns or column == outcome_column:
            continue
        if is_identifier_column(column):
            continue

        finding = finding_by_column.get(column, {})
        finding_type = str(finding.get("type", "")).lower()

        if finding_type != "sensitive":
            continue

        filtered.append(column)

    return filtered


def build_column_profile(df: pd.DataFrame, sample_size: int = 8) -> dict:
    profile = {}

    for col in df.columns:
        series = df[col].dropna()
        profile[col] = {
            "dtype": str(df[col].dtype),
            "sample_values": [json_safe(v) for v in series.head(sample_size).tolist()],
            "unique_count": int(series.nunique()),
            "missing_count": int(df[col].isna().sum()),
        }

    return profile


def infer_outcome_column(df: pd.DataFrame) -> str:
    preferred_names = [
        "ai_decision",
        "model_decision",
        "algorithm_decision",
        "prediction",
        "predicted_outcome",
        "decision_outcome",
        "final_decision",
        "outcome_label",
        "hired",
        "hire",
        "selected",
        "approved",
        "accepted",
        "admitted",
        "shortlisted",
        "outcome",
        "decision",
        "result",
        "label",
        "target",
    ]

    lower_to_actual = {normalize_column_name(c): c for c in df.columns}
    for name in preferred_names:
        if name in lower_to_actual:
            return lower_to_actual[name]

    for col in df.columns:
        key = normalize_column_name(col)
        if "override" in key:
            continue
        if key.endswith("_decision") or key.endswith("_outcome") or key.endswith("_label"):
            return col

    for col in reversed(df.columns.tolist()):
        key = normalize_column_name(col)
        if "override" in key or is_identifier_column(col):
            continue
        values = set(str(v).strip().lower() for v in df[col].dropna().unique()[:20])
        if values and values.issubset({"yes", "no", "true", "false", "1", "0", "hired", "rejected", "approved", "denied"}):
            return col

    return df.columns[-1] if len(df.columns) else ""


def infer_favorable_value(df: pd.DataFrame, outcome_column: str) -> str:
    if not outcome_column or outcome_column not in df.columns:
        return "Yes"

    values = [v for v in df[outcome_column].dropna().unique().tolist()]
    lowered = {str(v).strip().lower(): v for v in values}

    for candidate in ["yes", "true", "1", "hired", "approved", "accepted", "selected", "pass"]:
        if candidate in lowered:
            return str(lowered[candidate])

    if pd.api.types.is_numeric_dtype(df[outcome_column]) and values:
        return str(max(values))

    return str(values[0]) if values else "Yes"
