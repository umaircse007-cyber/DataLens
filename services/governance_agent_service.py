from typing import Any

PII_SEMANTIC_LABELS = {
    "Email Address",
    "Phone Number",
    "Person Name",
    "Date of Birth",
    "Demographic Attribute",
    "Geographic Attribute",
}

RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def _eu_clauses_from_profiles(profiles: list[dict]) -> list[dict]:
    clauses = []
    seen = set()
    for profile in profiles:
        flag = profile.get("fairness_flag") or {}
        article = (flag.get("eu_ai_act_article") or "").strip()
        if not article or article in seen:
            continue
        seen.add(article)
        clauses.append({
            "clause": article,
            "title": "Fairness & transparency review",
            "trigger_reason": flag.get("reason", f"Column {profile.get('column_name')} flagged for review."),
            "severity": "High Risk" if flag.get("is_sensitive") else "Limited Risk",
        })
    return clauses


def build_governance(
    column_dictionary: list[dict],
    profiles: list[dict],
    quality_audit: dict,
    eu_clauses: list[dict] | None = None,
) -> dict[str, Any]:
    detected_pii = []
    recommendations = []
    risk_level = "Low"
    merged_eu = list(eu_clauses or []) + _eu_clauses_from_profiles(profiles)

    for entry in column_dictionary:
        semantic = (entry.get("semantic_type") or {}).get("label", "")
        sensitivity = entry.get("sensitivity") or {}
        column = entry.get("column_name", "")
        if semantic in PII_SEMANTIC_LABELS or sensitivity.get("pii"):
            detected_pii.append(column)
        if sensitivity.get("level") == "High":
            risk_level = "High"
        elif sensitivity.get("level") == "Medium" and RISK_ORDER[risk_level] < RISK_ORDER["Medium"]:
            risk_level = "Medium"

    if not detected_pii:
        recommendations.append("No direct PII fields were detected, but review identifiers and free-text columns manually.")
    else:
        recommendations.append(
            f"Restrict access to PII fields ({', '.join(detected_pii[:5])}) and document lawful basis under GDPR."
        )

    critical = int(quality_audit.get("critical_count") or 0)
    if critical:
        risk_level = "High"
        recommendations.append("Resolve critical data quality findings before production use or model training.")

    recommendations.extend([
        "Apply role-based access controls for sensitive and proxy attributes.",
        "Define retention limits for person-level fields and audit deletion workflows.",
        "Map high-risk AI use cases to EU AI Act Articles 9–15 when deploying automated decisions.",
    ])

    flagged_column_count = sum(1 for profile in profiles if profile.get("fairness_flag"))
    if flagged_column_count:
        recommendations.append(
            f"Review {flagged_column_count} fairness-flagged column(s) for bias testing and documentation."
        )

    return {
        "detected_pii": detected_pii,
        "risk_level": risk_level,
        "recommendations": recommendations[:8],
        "gdpr_notes": [
            "Personal data requires a documented lawful basis and purpose limitation.",
            "Data subjects may have rights to access, rectification, and erasure where applicable.",
            "Cross-border transfers may require Standard Contractual Clauses or adequacy decisions.",
        ],
        "retention_suggestions": [
            "Identifiers and contact fields: retain only while the relationship is active plus statutory limits.",
            "Compensation and performance fields: align retention with HR policy and local labor law.",
            "Audit logs of processing: retain long enough to demonstrate compliance, then archive or delete.",
        ],
        "eu_ai_act": merged_eu[:12],
        "flagged_column_count": flagged_column_count,
    }
