from typing import Any


def build_dataset_story(
    metadata: dict,
    understanding: dict,
    health: dict,
    quality_audit: dict,
    column_dictionary: list[dict],
) -> dict[str, str]:
    filename = metadata.get("filename", "dataset")
    rows = metadata.get("row_count", 0)
    cols = metadata.get("column_count", 0)
    dataset_type = (understanding.get("dataset_type") or {}).get("label", "dataset")
    entity = (understanding.get("main_entity") or {}).get("label", "records")
    trust = (health.get("overall_trust") or {}).get("score", quality_audit.get("score", 0))
    top_issues = [
        finding.get("title", "")
        for finding in (quality_audit.get("top_findings") or quality_audit.get("findings") or [])[:4]
        if finding.get("title")
    ]
    issue_text = ", ".join(top_issues) if top_issues else "no major quality issues"

    semantic_highlights = [
        entry.get("column_name", "")
        for entry in column_dictionary
        if (entry.get("semantic_type") or {}).get("confidence", 0) >= 0.85
    ][:6]

    executive = (
        f"This dataset ({filename}) contains {rows:,} {entity.lower()} records across {cols} fields. "
        f"It is classified as a {dataset_type} with an overall trust score of {trust}/100. "
        f"Key dimensions include {', '.join(semantic_highlights[:4]) or 'the uploaded columns'}. "
        f"Detected issues include {issue_text}."
    )

    business = (
        f"{understanding.get('business_purpose', '')} "
        f"Suggested use cases: {', '.join(understanding.get('use_cases') or [])}. "
        f"The data is {'suitable for decision-making after minor cleanup' if trust >= 75 else 'usable with targeted remediation' if trust >= 55 else 'not yet trustworthy for high-stakes decisions'}."
    )

    technical = (
        f"Analysis covered {cols} columns and {rows:,} rows. "
        f"Health breakdown — completeness {(health.get('completeness') or {}).get('score', 'n/a')}, "
        f"consistency {(health.get('consistency') or {}).get('score', 'n/a')}, "
        f"validity {(health.get('validity') or {}).get('score', 'n/a')}, "
        f"uniqueness {(health.get('uniqueness') or {}).get('score', 'n/a')}, "
        f"governance {(health.get('governance') or {}).get('score', 'n/a')}. "
        f"Audit findings: {quality_audit.get('critical_count', 0)} critical/high, "
        f"{quality_audit.get('warning_count', 0)} warnings."
    )

    return {
        "executive_summary": executive.strip(),
        "business_summary": business.strip(),
        "technical_summary": technical.strip(),
    }
