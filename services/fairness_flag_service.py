import json
import os
import re

import pandas as pd

from services.dataset_service import correlation_gate, infer_favorable_value, infer_outcome_column
from services.security_service import build_safe_payload


SENSITIVE_HEURISTICS = {
    "gender": "Gender is a protected demographic attribute and must be reviewed for fairness and compliance.",
    "sex": "Sex is a protected demographic attribute and must be reviewed for fairness and compliance.",
    "age": "Age can be a protected attribute in employment, credit, insurance, and public-service contexts.",
    "dob": "Date of birth directly reveals age and can create protected-attribute exposure.",
    "date_of_birth": "Date of birth directly reveals age and can create protected-attribute exposure.",
    "birth_date": "Birth date directly reveals age and can create protected-attribute exposure.",
    "race": "Race is a protected attribute and requires explicit fairness review.",
    "ethnicity": "Ethnicity is a protected attribute and requires explicit fairness review.",
    "religion": "Religion is a protected attribute and requires explicit fairness review.",
    "disability": "Disability status is a protected attribute and requires explicit fairness review.",
}

PROXY_HEURISTICS = {
    "full_name": "Full names can encode gender, ethnicity, religion, or national origin signals.",
    "first_name": "First names can encode gender, ethnicity, religion, or national origin signals.",
    "last_name": "Last names can encode ethnicity, religion, or national origin signals.",
    "name": "Names can encode gender, ethnicity, religion, or national origin signals.",
    "zip": "Zip codes can proxy for geography, race, income, and socioeconomic background.",
    "zipcode": "Zip codes can proxy for geography, race, income, and socioeconomic background.",
    "postal": "Postal codes can proxy for geography, race, income, and socioeconomic background.",
    "postcode": "Postcodes can proxy for geography, race, income, and socioeconomic background.",
    "city": "City can proxy for geography, ethnicity, income, and neighborhood-level opportunity.",
    "college_tier": "College tier can proxy for socioeconomic background, historical access, and educational privilege.",
    "college": "College information can proxy for socioeconomic background and historical access.",
    "university": "University information can proxy for socioeconomic background and historical access.",
    "school": "School information can proxy for socioeconomic background and neighborhood opportunity.",
}


def _clean_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_findings(text: str) -> list[dict]:
    try:
        parsed = json.loads(_clean_json(text))
    except Exception:
        match = re.search(r"\[[\s\S]*\]", text or "")
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return []

    if isinstance(parsed, dict):
        parsed = parsed.get("columns", parsed.get("findings", []))
    return parsed if isinstance(parsed, list) else []


def _gemini_assess_columns(payloads: list[dict]) -> list[dict]:
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return []

    prompt = f"""
You are DataLens, a fairness audit reviewer.

Analyze the safe per-column payloads below. Identify columns that are direct sensitive attributes or plausible proxies for protected attributes. Do not flag legitimate merit-based columns unless they are clearly hidden proxies.

Safe column payloads:
{json.dumps(payloads, indent=2, default=str)}

Return ONLY valid JSON:
[
  {{
    "column": "Column Name",
    "is_sensitive": true,
    "is_proxy": false,
    "reason": "Short reason",
    "confidence": "High"
  }}
]
"""

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        model_names = [
            (os.environ.get("GEMINI_MODEL") or "").replace("models/", "").strip(),
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
        ]
        for model_name in [name for name in model_names if name]:
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                findings = _parse_findings(getattr(response, "text", ""))
                if findings:
                    return findings
            except Exception:
                continue
    except Exception:
        return []
    return []


def _groq_verify_findings(payloads: list[dict], proposed_findings: list[dict]) -> dict[str, dict]:
    from services.groq_client import groq_chat, is_groq_configured
    from services.groq_keys import ensure_env_loaded

    ensure_env_loaded()
    if not proposed_findings:
        return {}
    if not is_groq_configured():
        return {}

    safe_findings = [
        {
            "column": item.get("column"),
            "is_sensitive": bool(item.get("is_sensitive")),
            "is_proxy": bool(item.get("is_proxy")),
            "reason": item.get("reason"),
            "confidence": item.get("confidence", "Medium"),
        }
        for item in proposed_findings
        if item.get("column")
    ]
    prompt = f"""
You are Groq acting as an independent fairness verification reviewer for DataLens.

Review the proposed sensitive/proxy flags below using only the safe column payloads and proposed reasons.
Do not use full rows, do not infer beyond these payloads, and be conservative:
- Confirm if the proposed flag is plausible and useful for compliance review.
- Challenge if the flag looks unsupported or is likely a legitimate business feature.
- Mark uncertain if the safe metadata is insufficient.

Safe column payloads:
{json.dumps(payloads, indent=2, default=str)}

Proposed flags from Gemini/heuristics:
{json.dumps(safe_findings, indent=2, default=str)}

Return ONLY valid JSON:
[
  {{
    "column": "Column Name",
    "verdict": "Confirmed|Challenged|Uncertain",
    "reason": "Short verification explanation",
    "confidence": "High|Medium|Low"
  }}
]
"""

    text, error = groq_chat(
        [
            {"role": "system", "content": "You are an independent fairness verification reviewer. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=2000,
    )
    if text:
        items = _parse_findings(text)
        if items:
            return {
                str(item.get("column")): {
                    "verdict": str(item.get("verdict", "Uncertain")),
                    "reason": str(item.get("reason", "Groq reviewed the proposed fairness flag.")),
                    "confidence": str(item.get("confidence", "Medium")),
                }
                for item in items
                if item.get("column")
            }
    if error:
        import logging
        logging.getLogger("datalens.fairness").warning("Groq fairness verification failed: %s", error)
    return {}


def _article_for(finding: dict) -> str | None:
    if bool(finding.get("is_sensitive")):
        return "Article 10(2)(f)"
    if bool(finding.get("is_proxy")):
        return "Article 13"
    return None


def _normalized_column(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def _heuristic_assess_columns(profiles: list[dict]) -> list[dict]:
    findings = []
    for profile in profiles:
        column = str(profile.get("column_name", ""))
        key = _normalized_column(column)
        sensitive_reason = next((reason for hint, reason in SENSITIVE_HEURISTICS.items() if hint == key or hint in key), None)
        proxy_reason = None if sensitive_reason else next(
            (reason for hint, reason in PROXY_HEURISTICS.items() if hint == key or hint in key),
            None,
        )
        if sensitive_reason or proxy_reason:
            findings.append({
                "column": column,
                "is_sensitive": bool(sensitive_reason),
                "is_proxy": bool(proxy_reason),
                "reason": sensitive_reason or proxy_reason,
                "confidence": "High" if sensitive_reason else "Medium",
            })
    return findings


def flag_sensitive_columns(profiles: list[dict], df: pd.DataFrame) -> list[dict]:
    from services.groq_keys import ensure_env_loaded

    ensure_env_loaded()
    payloads = [
        build_safe_payload(df, profile["column_name"])
        for profile in profiles
        if profile.get("column_name") in df.columns
    ]
    findings = _gemini_assess_columns(payloads)
    heuristic_findings = _heuristic_assess_columns(profiles)
    finding_by_column = {str(item.get("column")): item for item in findings if item.get("column")}
    for item in heuristic_findings:
        existing = finding_by_column.get(str(item.get("column")))
        if not existing:
            finding_by_column[str(item.get("column"))] = item
            continue
        existing["is_sensitive"] = bool(existing.get("is_sensitive")) or bool(item.get("is_sensitive"))
        existing["is_proxy"] = bool(existing.get("is_proxy")) or bool(item.get("is_proxy"))
        existing["reason"] = f"{existing.get('reason', '')} {item.get('reason', '')}".strip()
        if item.get("confidence") == "High":
            existing["confidence"] = "High"

    proposed_findings = list(finding_by_column.values())
    groq_verifications = _groq_verify_findings(payloads, proposed_findings)

    outcome_column = infer_outcome_column(df)
    outcome_identified = bool(outcome_column and outcome_column in df.columns)
    favorable_value = infer_favorable_value(df, outcome_column) if outcome_identified else None

    for profile in profiles:
        column = profile.get("column_name")
        finding = finding_by_column.get(str(column))
        if not finding or not (finding.get("is_sensitive") or finding.get("is_proxy")):
            continue
        if outcome_identified and column == outcome_column:
            continue

        reason = str(finding.get("reason", "Potential fairness-relevant column."))
        from services.groq_client import is_groq_configured

        if str(column) in groq_verifications:
            groq_verification = groq_verifications[str(column)]
        elif is_groq_configured():
            groq_verification = {
                "verdict": "Pending",
                "reason": "Groq is configured but did not return a verification for this column in this run.",
                "confidence": "Low",
            }
        else:
            groq_verification = {
                "verdict": "Not configured",
                "reason": "Set GROQ_API_KEY in your .env file (copy from .env.example) and re-analyse the dataset.",
                "confidence": "Low",
            }
        gate_passes = True
        if outcome_identified:
            stats = correlation_gate(df, column, outcome_column, favorable_value)
            gate_passes = bool(stats["passes"])
            reason = f"{reason} Correlation with outcome: r={stats['r']:.3f}, p={stats['p']:.3f}."
        else:
            reason = f"{reason} Outcome column not identified."

        if not gate_passes:
            profile["fairness_flag"] = {
                "is_sensitive": bool(finding.get("is_sensitive")),
                "is_proxy": bool(finding.get("is_proxy")),
                "reason": f"{reason} Correlation gate did not pass, so treat this as a review warning rather than evidence of outcome-linked bias.",
                "eu_ai_act_article": _article_for(finding),
                "confidence": "Medium",
                "review_only": True,
                "groq_verification": groq_verification,
                "verification_status": groq_verification.get("verdict", "Uncertain"),
            }
            continue

        profile["fairness_flag"] = {
            "is_sensitive": bool(finding.get("is_sensitive")),
            "is_proxy": bool(finding.get("is_proxy")),
            "reason": reason,
            "eu_ai_act_article": _article_for(finding),
            "confidence": "High" if str(finding.get("confidence", "")).lower() == "high" else "Medium",
            "groq_verification": groq_verification,
            "verification_status": groq_verification.get("verdict", "Uncertain"),
        }

    return profiles
