from typing import Any


def _dimension(score: int, reasoning: str, evidence: list[str]) -> dict[str, Any]:
    return {
        "score": int(max(0, min(100, score))),
        "reasoning": reasoning,
        "evidence": evidence[:8],
    }


def build_health_scores(
    profiles: list[dict],
    quality_audit: dict,
    governance: dict,
    column_dictionary: list[dict],
) -> dict[str, Any]:
    row_count = max(1, int((quality_audit.get("findings") or [{}])[0].get("row_count") or 1))
    avg_null = sum(float(profile.get("null_pct") or 0) for profile in profiles) / max(len(profiles), 1)
    completeness = _dimension(
        int(max(0, 100 - avg_null * 1.5)),
        "Based on average null rates across all columns.",
        [f"{profile.get('column_name')}: {profile.get('null_pct')}% null" for profile in profiles if profile.get("null_pct")][:6],
    )

    consistency_findings = [
        item for item in (quality_audit.get("findings") or [])
        if item.get("category") in {"Semantic Consistency", "Standardization", "Business Logic"}
    ]
    consistency = _dimension(
        max(0, 100 - len(consistency_findings) * 12),
        "Reflects semantic mismatches, standardization gaps, and business-rule conflicts.",
        [item.get("title", "") for item in consistency_findings[:6]],
    )

    validity_findings = [
        item for item in (quality_audit.get("findings") or [])
        if item.get("category") in {"Validity", "Outliers"}
    ]
    validity = _dimension(
        max(0, 100 - len(validity_findings) * 10),
        "Reflects invalid formats, impossible values, and extreme outliers.",
        [item.get("title", "") for item in validity_findings[:6]],
    )

    duplicate_findings = [
        item for item in (quality_audit.get("findings") or [])
        if item.get("category") == "Duplicates"
    ]
    uniqueness = _dimension(
        max(0, 100 - len(duplicate_findings) * 15),
        "Reflects exact and fuzzy duplicate identity patterns.",
        [item.get("title", "") for item in duplicate_findings[:6]],
    )

    pii_count = len(governance.get("detected_pii") or [])
    flagged = int(governance.get("flagged_column_count") or 0)
    governance_score = _dimension(
        max(0, 100 - pii_count * 8 - flagged * 5),
        "Reflects PII exposure and sensitive or proxy attributes requiring governance review.",
        (governance.get("detected_pii") or [])[:6],
    )

    documented = sum(
        1 for entry in column_dictionary
        if len((entry.get("business_description") or "").split()) >= 8
    )
    documentation = _dimension(
        int((documented / max(len(column_dictionary), 1)) * 100),
        "Measures how many columns have substantive business descriptions.",
        [
            entry.get("column_name", "")
            for entry in column_dictionary
            if len((entry.get("business_description") or "").split()) >= 8
        ][:6],
    )

    weights = {
        "completeness": 0.2,
        "consistency": 0.2,
        "validity": 0.2,
        "uniqueness": 0.15,
        "governance": 0.15,
        "documentation_quality": 0.1,
    }
    dimensions = {
        "completeness": completeness,
        "consistency": consistency,
        "validity": validity,
        "uniqueness": uniqueness,
        "governance": governance_score,
        "documentation_quality": documentation,
    }
    overall = int(round(sum(dimensions[key]["score"] * weight for key, weight in weights.items())))

    return {
        **dimensions,
        "overall_trust": _dimension(
            overall,
            "Weighted composite of completeness, consistency, validity, uniqueness, governance, and documentation quality.",
            [f"{name}: {dimensions[name]['score']}" for name in weights],
        ),
    }
