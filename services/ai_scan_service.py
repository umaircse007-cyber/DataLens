import asyncio
import json
import os
import re
from typing import Any

import pandas as pd

from services.security_service import build_safe_payload


SYSTEM_PROMPT = "You are a data documentation expert."


def _clean_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_json_dict(text: str) -> dict | None:
    try:
        parsed = json.loads(_clean_json(text))
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text or "")
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return None

    return parsed if isinstance(parsed, dict) else None


def _column_prompt(payload: dict) -> str:
    return f"""
{SYSTEM_PROMPT}

Describe this dataset column using only the metadata below. Do not infer from full rows.

Safe column payload:
{json.dumps(payload, indent=2, default=str)}

Return ONLY valid JSON with this shape:
{{
  "display_name": "Human readable name",
  "description": "One plain-English sentence describing what the column likely represents.",
  "domain": "Business domain or Unknown",
  "data_quality_flag": "None|Missing values|High cardinality|Outliers|Ambiguous meaning|Other",
  "data_quality_note": "Short note or empty string"
}}
"""


def _candidate_gemini_models(client: Any) -> list[str]:
    names = [
        (os.environ.get("GEMINI_MODEL") or "").replace("models/", "").strip(),
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
    ]
    return [name for name in names if name]


def _describe_column_gemini_sync(payload: dict) -> dict | None:
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        prompt = _column_prompt(payload)
        for model_name in _candidate_gemini_models(client):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                parsed = _parse_json_dict(getattr(response, "text", ""))
                if parsed:
                    return parsed
            except Exception:
                continue
    except Exception:
        return None
    return None


def _describe_column_groq_sync(payload: dict) -> dict | None:
    from services.groq_client import groq_chat, is_groq_configured
    from services.groq_keys import ensure_env_loaded

    ensure_env_loaded()
    if not is_groq_configured():
        return None

    prompt = _column_prompt(payload)
    text, _error = groq_chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    if not text:
        return None
    parsed = _parse_json_dict(text)
    return parsed if parsed else None


async def describe_column_gemini(payload: dict) -> dict | None:
    return await asyncio.to_thread(_describe_column_gemini_sync, payload)


async def describe_column_groq(payload: dict) -> dict | None:
    return await asyncio.to_thread(_describe_column_groq_sync, payload)


def _words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())
        if len(word) > 2
    }


def consensus_description(gemini_result: dict | None, groq_result: dict | None) -> dict:
    if gemini_result and groq_result:
        gemini_words = _words(str(gemini_result.get("description", "")))
        groq_words = _words(str(groq_result.get("description", "")))
        overlap = len(gemini_words & groq_words) / max(len(gemini_words), 1)
        confidence = "Confirmed" if overlap > 0.4 else "Review needed"
        
        disagreement_reason = None
        if confidence == "Review needed":
            disagreement_reason = f"Gemini: {gemini_result.get('description')} | Groq: {groq_result.get('description')}"
        
        return {
            "display_name": gemini_result.get("display_name"),
            "description": gemini_result.get("description"),
            "domain": gemini_result.get("domain"),
            "data_quality_flag": gemini_result.get("data_quality_flag", "None"),
            "data_quality_note": gemini_result.get("data_quality_note", ""),
            "confidence": confidence,
            "groq_description": groq_result.get("description"),
            "disagreement_reason": disagreement_reason,
        }

    result = gemini_result or groq_result
    if result:
        return {
            "display_name": result.get("display_name"),
            "description": result.get("description"),
            "domain": result.get("domain"),
            "data_quality_flag": result.get("data_quality_flag", "None"),
            "data_quality_note": result.get("data_quality_note", ""),
            "confidence": "Single model",
            "groq_description": groq_result.get("description") if groq_result else None,
            "disagreement_reason": None,
        }

    return {
        "display_name": None,
        "description": None,
        "domain": None,
        "data_quality_flag": "None",
        "data_quality_note": "",
        "confidence": "Fallback",
        "groq_description": None,
        "disagreement_reason": None,
    }


def _normalized_column(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def _description_too_generic(description: str | None, column: str) -> bool:
    text = (description or "").strip().lower()
    if not text:
        return True
    generic_phrases = [
        "contains names",
        "represents a date of birth",
        "contains the names",
        "specific condition",
        "represents the",
    ]
    return len(text.split()) < 8 or any(phrase in text for phrase in generic_phrases)


def _apply_governance_description_hints(profile: dict) -> None:
    column = str(profile.get("column_name", ""))
    key = _normalized_column(column)
    hints = {
        "full_name": (
            "Full Name",
            "Person name field that may identify individuals and can encode gender, ethnicity, religion, or national origin signals.",
        ),
        "first_name": (
            "First Name",
            "Given-name field that may identify individuals and can encode gender, ethnicity, religion, or national origin signals.",
        ),
        "last_name": (
            "Last Name",
            "Surname field that may identify individuals and can encode ethnicity, religion, or national origin signals.",
        ),
        "date_of_birth": (
            "Date of Birth",
            "Birth date field that directly reveals age and may be redundant with an explicit age column.",
        ),
        "dob": (
            "Date of Birth",
            "Birth date field that directly reveals age and may be redundant with an explicit age column.",
        ),
        "city": (
            "City",
            "Geographic location field that may act as a proxy for neighborhood, income, ethnicity, or opportunity patterns.",
        ),
        "age": (
            "Age",
            "Person age field that can be fairness-sensitive and may overlap with date-of-birth information.",
        ),
        "gender": (
            "Gender",
            "Demographic gender field that requires careful handling for fairness, privacy, and compliance review.",
        ),
        "zip_code": (
            "Zip Code",
            "Postal geography field that may act as a proxy for income, race, neighborhood, and socioeconomic background.",
        ),
        "college_tier": (
            "College Tier",
            "Education prestige grouping that may proxy for socioeconomic background, historical access, and institutional privilege.",
        ),
    }
    match = next((value for hint, value in hints.items() if hint == key or hint in key), None)
    if not match:
        return
    display_name, description = match
    profile["display_name"] = profile.get("display_name") or display_name
    if _description_too_generic(profile.get("description"), column):
        profile["description"] = description


async def scan_all_columns(df: pd.DataFrame, profiles: list[dict]) -> list[dict]:
    for profile in profiles:
        column = profile.get("column_name")
        if column not in df.columns:
            continue
        payload = build_safe_payload(df, column)
        gemini_result, groq_result = await asyncio.gather(
            describe_column_gemini(payload),
            describe_column_groq(payload),
        )
        profile.update(consensus_description(gemini_result, groq_result))
        _apply_governance_description_hints(profile)
    return profiles
