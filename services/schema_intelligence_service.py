import re
from typing import Any

import pandas as pd

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^\+?[\d\s().-]{10,18}$")
URL_RE = re.compile(r"^https?://[^\s]+$", re.I)

NAME_HINTS = {
    "customer_id": "Primary Key Candidate",
    "user_id": "Primary Key Candidate",
    "employee_id": "Primary Key Candidate",
    "id": "Identifier",
    "uuid": "Identifier",
    "email": "Email Address",
    "e_mail": "Email Address",
    "phone": "Phone Number",
    "mobile": "Phone Number",
    "url": "Web URL",
    "website": "Web URL",
    "country": "Geographic Attribute",
    "nation": "Geographic Attribute",
    "city": "Geographic Attribute",
    "state": "Geographic Attribute",
    "zip": "Geographic Attribute",
    "postal": "Geographic Attribute",
    "salary": "Compensation Metric",
    "income": "Compensation Metric",
    "compensation": "Compensation Metric",
    "pay": "Compensation Metric",
    "currency": "Currency Code",
    "registration_date": "Event Date",
    "hire_date": "Event Date",
    "order_date": "Event Date",
    "created_at": "Event Date",
    "dob": "Date of Birth",
    "date_of_birth": "Date of Birth",
    "birth_date": "Date of Birth",
    "birthdate": "Date of Birth",
    "age": "Demographic Attribute",
    "gender": "Demographic Attribute",
    "sex": "Demographic Attribute",
    "race": "Demographic Attribute",
    "ethnicity": "Demographic Attribute",
    "department": "Organizational Attribute",
    "full_name": "Person Name",
    "first_name": "Person Name",
    "last_name": "Person Name",
    "name": "Person Name",
}


def _norm_column(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def _inference(label: str, confidence: float, evidence: list[str]) -> dict[str, Any]:
    return {
        "label": label,
        "confidence": round(min(max(confidence, 0.0), 1.0), 2),
        "evidence": evidence,
    }


def _value_pattern_type(series: pd.Series) -> tuple[str | None, float, list[str]]:
    sample = series.dropna().astype(str).str.strip().head(200)
    if sample.empty:
        return None, 0.0, []

    email_hits = sum(1 for value in sample if EMAIL_RE.match(value))
    if email_hits / len(sample) >= 0.8:
        return "Email Address", 0.95, [f"{email_hits}/{len(sample)} values match email pattern"]

    phone_digits = []
    for value in sample:
        digits = re.sub(r"\D", "", value)
        if digits and 10 <= len(digits) <= 15:
            phone_digits.append(value)
    if len(phone_digits) / len(sample) >= 0.75:
        return "Phone Number", 0.9, [f"{len(phone_digits)}/{len(sample)} values look like phone numbers"]

    url_hits = sum(1 for value in sample if URL_RE.match(value))
    if url_hits / len(sample) >= 0.7:
        return "Web URL", 0.88, [f"{url_hits}/{len(sample)} values match URL pattern"]

    return None, 0.0, []


def infer_semantic_type(column: str, series: pd.Series, row_count: int) -> dict[str, Any]:
    key = _norm_column(column)
    evidence: list[str] = []

    for hint, label in NAME_HINTS.items():
        if hint == key or hint in key:
            evidence.append(f"Column name matches '{hint}'")
            if label == "Primary Key Candidate":
                non_null = series.dropna()
                unique_ratio = non_null.nunique() / max(len(non_null), 1)
                if row_count and unique_ratio >= 0.98 and float(series.isna().mean()) < 0.02:
                    evidence.append("Near-unique non-null values across all rows")
                    return _inference(label, 0.94, evidence)
                return _inference("Identifier", 0.78, evidence + ["High-cardinality identifier field"])
            return _inference(label, 0.86, evidence)

    pattern_label, pattern_conf, pattern_evidence = _value_pattern_type(series)
    if pattern_label:
        return _inference(pattern_label, pattern_conf, pattern_evidence)

    non_null = series.dropna()
    if row_count and len(non_null) == row_count and non_null.nunique() == row_count:
        return _inference(
            "Primary Key Candidate",
            0.92,
            ["All values are unique and non-null"],
        )

    if pd.api.types.is_datetime64_any_dtype(series):
        return _inference("Event Date", 0.8, ["Pandas datetime dtype"])

    if pd.api.types.is_numeric_dtype(series):
        if any(token in key for token in ("amount", "price", "cost", "revenue", "fee")):
            return _inference("Financial Measure", 0.82, ["Numeric column with financial naming"])
        return _inference("Numeric Measure", 0.65, ["Numeric dtype without stronger semantic hint"])

    if non_null.nunique() <= max(20, int(row_count * 0.05)):
        return _inference("Categorical Attribute", 0.7, ["Low cardinality relative to row count"])

    return _inference("General Attribute", 0.55, ["No strong semantic pattern detected"])


def build_schema_intelligence(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    row_count = len(df)
    return {
        str(column): infer_semantic_type(str(column), df[column], row_count)
        for column in df.columns
    }
