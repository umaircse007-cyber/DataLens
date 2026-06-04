import pandas as pd


def _add_deduction(deductions: list[dict], reason: str, points: int) -> None:
    if points > 0:
        deductions.append({"reason": reason, "points": int(points)})


def _grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


GRADE_SCALE = [
    {"grade": "A", "range": "90-100", "label": "Production-ready"},
    {"grade": "B", "range": "80-89", "label": "Good"},
    {"grade": "C", "range": "70-79", "label": "Acceptable"},
    {"grade": "D", "range": "60-69", "label": "Needs work"},
    {"grade": "F", "range": "Below 60", "label": "Poor candidate for AI or analytics use"},
]


def _impact(points: int) -> str:
    if points >= 14:
        return "High"
    if points >= 7:
        return "Moderate"
    return "Low"


def _factor(title: str, detected: list[str], impact: str, readiness_impact: str) -> dict:
    return {
        "title": title,
        "detected": [item for item in detected if item],
        "impact": impact,
        "readiness_impact": readiness_impact,
    }


def ml_readiness_score(
    df: pd.DataFrame,
    profiles: list[dict],
    quality_audit: dict | None = None,
    governance: dict | None = None,
) -> dict:
    deductions: list[dict] = []
    factors: list[dict] = []

    avg_null_rate = sum(float(profile.get("null_pct") or 0) for profile in profiles) / max(len(profiles), 1)
    if avg_null_rate > 10:
        _add_deduction(deductions, "Average null rate is above 10%.", min(20, round(avg_null_rate / 100 * 20)))

    elevated_null_columns = [
        profile for profile in profiles
        if float(profile.get("null_pct") or 0) >= 5
    ]
    _add_deduction(
        deductions,
        "One or more columns have elevated null rates that may skew analysis.",
        min(10, len(elevated_null_columns) * 2),
    )

    high_cardinality = [
        profile
        for profile in profiles
        if not str(profile.get("dtype", "")).startswith(("int", "float"))
        and int(profile.get("unique_count") or 0) > 100
    ]
    _add_deduction(
        deductions,
        "High-cardinality categorical columns may need encoding or grouping.",
        min(15, len(high_cardinality) * 5),
    )

    flagged_count = sum(1 for profile in profiles if profile.get("fairness_flag"))
    flagged_columns = [profile.get("column_name") for profile in profiles if profile.get("fairness_flag")]
    _add_deduction(
        deductions,
        "Sensitive or proxy columns require fairness review before modeling.",
        min(20, flagged_count * 10),
    )
    if flagged_columns:
        factors.append(_factor(
            "Fairness Review Recommended",
            flagged_columns,
            "These attributes may influence model outcomes and should be reviewed for bias and fairness before use in predictive modeling.",
            _impact(min(20, flagged_count * 10)),
        ))

    duplicate_rate = float(df.duplicated().mean() * 100) if len(df) else 0.0
    if duplicate_rate > 5:
        _add_deduction(deductions, "Duplicate rows exceed 5%.", min(10, round(duplicate_rate / 100 * 10)))

    if len(df) < 500:
        _add_deduction(deductions, "Dataset has fewer than 500 rows.", 15)
        factors.append(_factor(
            "Dataset Size Review",
            [f"Rows: {len(df)}"],
            "The dataset may be too small for robust statistical analysis or machine learning generalization.",
            "Moderate",
        ))
    elif len(df) < 1000:
        _add_deduction(deductions, "Dataset has fewer than 1000 rows.", 7)
        factors.append(_factor(
            "Dataset Size Review",
            [f"Rows: {len(df)}"],
            "The dataset is usable for exploration but may still be small for reliable model generalization.",
            "Low",
        ))

    outlier_columns = 0
    for column in df.select_dtypes(include="number").columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(series) < 2:
            continue
        std = series.std()
        if not std:
            continue
        outlier_rate = float(((series - series.mean()).abs() > (3 * std)).mean() * 100)
        if outlier_rate > 5:
            outlier_columns += 1
    _add_deduction(
        deductions,
        "Numeric columns contain more than 5% extreme outliers.",
        min(10, outlier_columns * 5),
    )

    audit_findings = (quality_audit or {}).get("findings") or []
    quality_titles = [
        item.get("title", "")
        for item in audit_findings
        if item.get("category") in {"Validity", "Outliers", "Duplicates", "Business Logic", "Semantic Consistency", "Standardization"}
    ][:8]
    if quality_titles:
        factors.append(_factor(
            "Data Quality Issues",
            quality_titles,
            "Data quality issues may affect reliability of analytical or predictive outputs.",
            "High" if any((item.get("severity") in {"Critical", "High"}) for item in audit_findings) else "Moderate",
        ))

    gov = governance or {}
    pii_fields = gov.get("detected_pii") or []
    if pii_fields or flagged_columns:
        factors.append(_factor(
            "Governance Review",
            [f"PII: {field}" for field in pii_fields[:6]] + [f"Sensitive/proxy: {field}" for field in flagged_columns[:6]],
            "Additional transparency, access control, retention, and governance controls may be required.",
            "Moderate",
        ))

    score = max(0, 100 - sum(item["points"] for item in deductions))
    grade = _grade(score)
    if score >= 80:
        summary = "Good candidate for responsible analytics, with limited review needed before advanced use."
    elif score >= 60:
        summary = "Needs work before responsible AI deployment or advanced analytics."
    else:
        summary = "Poor candidate for AI or analytics use until major quality and governance issues are resolved."

    return {
        "label": "AI & Analytics Readiness",
        "score": int(score),
        "grade": grade,
        "grade_scale": GRADE_SCALE,
        "deductions": [{"reason": item["reason"]} for item in deductions],
        "factors": factors,
        "summary": summary,
    }
