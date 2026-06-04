import re
from typing import Any


def _norm(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def _sensitivity_from_profile(profile: dict, semantic: dict) -> dict[str, Any]:
    flag = profile.get("fairness_flag") or {}
    label = (semantic or {}).get("label", "")
    pii_labels = {
        "Email Address",
        "Phone Number",
        "Person Name",
        "Date of Birth",
        "Demographic Attribute",
    }
    is_pii = label in pii_labels or bool(flag.get("is_sensitive"))
    level = "High" if is_pii or flag.get("is_sensitive") else "Medium" if flag.get("is_proxy") else "Low"
    categories = []
    if flag.get("is_sensitive"):
        categories.append("Sensitive attribute")
    if flag.get("is_proxy"):
        categories.append("Proxy attribute")
    if label in pii_labels:
        categories.append("PII")
    return {
        "level": level,
        "pii": is_pii,
        "categories": categories,
    }


def _quality_notes_for_column(column: str, findings: list[dict], outliers: list[dict]) -> list[str]:
    notes = []
    key = _norm(column)
    for finding in findings:
        evidence = finding.get("evidence") or []
        if any(key in _norm(str(item)) or column in str(item) for item in evidence):
            notes.append(finding.get("summary") or finding.get("title", ""))
        elif key in _norm(finding.get("title", "")):
            notes.append(finding.get("summary") or finding.get("title", ""))
    for outlier in outliers:
        if outlier.get("column") == column:
            notes.append(outlier.get("explanation", ""))
    return [note for note in dict.fromkeys(notes) if note][:6]


def _business_layers(column: str, semantic: dict, profile: dict) -> tuple[str, str]:
    key = _norm(column)
    label = (semantic or {}).get("label", "")
    if "salary" in key or "compensation" in key or label == "Compensation Metric":
        return "Employee compensation or pay-related value.", "Payroll analysis, workforce planning, compensation benchmarking."
    if "email" in key or label == "Email Address":
        return "Contact address for a person or account.", "Identity matching, communication workflows, account verification."
    if "phone" in key or label == "Phone Number":
        return "Phone contact number for a person or account.", "Customer or employee contact validation and outreach."
    if "age" == key or label == "Demographic Attribute":
        return "Demographic attribute describing a person.", "Fairness review, demographic analysis, eligibility or segmentation checks."
    if "birth" in key or "dob" in key or label == "Date of Birth":
        return "Birth date used to derive age and demographic context.", "Age calculation, identity verification, fairness and privacy review."
    if "name" in key or label == "Person Name":
        return "Person identity field.", "Identity matching, duplicate review, record linkage."
    if "country" in key or label == "Geographic Attribute":
        return "Geographic location attribute.", "Regional reporting, compliance checks, localization and segmentation."
    if "currency" in key or label == "Currency Code":
        return "Currency used for monetary values.", "Financial normalization, regional reporting, country-currency validation."
    if "department" in key or label == "Organizational Attribute":
        return "Organizational grouping or business unit.", "Workforce reporting, performance comparison, cost-center analysis."
    if label in {"Primary Key Candidate", "Identifier"}:
        return "Unique or near-unique record identifier.", "Deduplication, joins, lineage tracking, record lookup."
    if label == "Numeric Measure":
        return "Quantitative measurement or score.", "Trend analysis, benchmarking, distribution review."
    return profile.get("description") or "Business attribute captured in the uploaded dataset.", "Exploratory analysis, filtering, reporting, and data quality review."


def build_column_dictionary(
    profiles: list[dict],
    schema_map: dict[str, dict],
    quality_audit: dict,
    outliers: list[dict],
) -> list[dict[str, Any]]:
    findings = quality_audit.get("findings") or []
    columns = []

    for profile in profiles:
        name = str(profile.get("column_name", ""))
        semantic = schema_map.get(name, {})
        sensitivity = _sensitivity_from_profile(profile, semantic)
        business_meaning, typical_use = _business_layers(name, semantic, profile)
        desc_conf = profile.get("confidence")
        if desc_conf == "Confirmed":
            confidence = 0.95
        elif desc_conf == "Review needed":
            confidence = 0.72
        elif desc_conf == "Fallback":
            confidence = 0.55
        else:
            confidence = float(semantic.get("confidence") or 0.6)

        columns.append({
            "column_name": name,
            "display_name": profile.get("display_name") or name,
            "technical_type": profile.get("dtype"),
            "semantic_type": semantic,
            "business_description": profile.get("description") or "",
            "business_meaning": business_meaning,
            "typical_use": typical_use,
            "example_values": (profile.get("sample_values") or profile.get("top_values") or [])[:5],
            "quality_notes": _quality_notes_for_column(name, findings, outliers),
            "sensitivity": sensitivity,
            "confidence": round(confidence, 2),
            "null_pct": profile.get("null_pct"),
            "unique_count": profile.get("unique_count"),
            "anomaly_note": profile.get("anomaly_note"),
            "fairness_flag": profile.get("fairness_flag"),
            "data_quality_flag": profile.get("data_quality_flag"),
        })

    return columns
