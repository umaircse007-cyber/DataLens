import json
import os
import re


def _clean_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_query_list(text: str) -> list[dict]:
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
        parsed = parsed.get("questions", [])
    if not isinstance(parsed, list):
        return []

    results = []
    for item in parsed[:5]:
        if not isinstance(item, dict):
            continue
        results.append({
            "question": str(item.get("question", "")),
            "pandas_query": str(item.get("pandas_query", "")),
            "sql_query": str(item.get("sql_query", "")),
        })
    return results


def _safe_column_name(profile: dict) -> str:
    return str(profile.get("column_name") or profile.get("display_name") or "").strip()


def _identifier_columns(profiles: list[dict]) -> list[str]:
    ids = []
    for profile in profiles:
        column = _safe_column_name(profile)
        semantic = (profile.get("semantic_type") or {}).get("label", "")
        if "Identifier" in semantic or column.lower().endswith("_id") or column.lower() == "id":
            ids.append(column)
    return ids


def _sensitive_columns(profiles: list[dict]) -> list[str]:
    sensitive = []
    for profile in profiles:
        column = _safe_column_name(profile)
        sensitivity = profile.get("sensitivity") or {}
        if sensitivity.get("pii") or sensitivity.get("level") in {"Medium", "High"} or profile.get("fairness_flag"):
            sensitive.append(column)
    return sensitive


def _fallback_queries(profiles: list[dict], context: dict | None = None) -> list[dict]:
    context = context or {}
    dataset = context.get("dataset") or {}
    readiness = context.get("readiness") or {}
    health = context.get("health") or {}
    quality_audit = context.get("quality_audit") or {}
    dataset_type = (dataset.get("dataset_type") or {}).get("label", "this dataset")
    entity = (dataset.get("main_entity") or {}).get("label", "records")
    identifiers = _identifier_columns(profiles)
    sensitive = _sensitive_columns(profiles)
    columns = [_safe_column_name(profile) for profile in profiles if _safe_column_name(profile)]
    first_col = columns[0] if columns else "*"
    id_col = identifiers[0] if identifiers else first_col
    sensitive_col = sensitive[0] if sensitive else first_col

    questions = [
        {
            "question": f"What does this {dataset_type.lower()} contain?",
            "pandas_query": "df.info()",
            "sql_query": "SELECT COUNT(*) AS row_count FROM table_name;",
        },
        {
            "question": f"Which columns describe the main {entity.lower()} entity?",
            "pandas_query": "df.columns.tolist()",
            "sql_query": "SELECT column_name FROM information_schema.columns WHERE table_name = 'table_name';",
        },
        {
            "question": "What are the biggest quality issues?",
            "pandas_query": "df.isna().mean().sort_values(ascending=False).head(10)",
            "sql_query": "SELECT * FROM table_name LIMIT 10;",
        },
        {
            "question": "Which fields contain sensitive or personal information?",
            "pandas_query": f"df[[{', '.join(repr(c) for c in sensitive[:5])}]].head()" if sensitive else "df.head()",
            "sql_query": f"SELECT {sensitive_col} FROM table_name LIMIT 10;",
        },
        {
            "question": f"Why is the AI & Analytics Readiness score {readiness.get('score', 'what it is')}?",
            "pandas_query": "df.describe(include='all')",
            "sql_query": "SELECT COUNT(*) AS rows_to_review FROM table_name;",
        },
        {
            "question": f"Why is the overall trust score {(health.get('overall_trust') or {}).get('score', 'what it is')}?",
            "pandas_query": "df.isna().sum()",
            "sql_query": "SELECT COUNT(*) AS total_records FROM table_name;",
        },
        {
            "question": "Which columns require governance review?",
            "pandas_query": f"df[[{', '.join(repr(c) for c in sensitive[:5])}]].head()" if sensitive else "df.head()",
            "sql_query": f"SELECT {sensitive_col} FROM table_name LIMIT 10;",
        },
        {
            "question": "Which columns are likely identifiers?",
            "pandas_query": f"df['{id_col}'].nunique()" if id_col != "*" else "df.nunique().sort_values(ascending=False).head()",
            "sql_query": f"SELECT COUNT(DISTINCT {id_col}) AS unique_values FROM table_name;" if id_col != "*" else "SELECT COUNT(*) FROM table_name;",
        },
        {
            "question": "How should this dataset be cleaned?",
            "pandas_query": "df.drop_duplicates().isna().sum()",
            "sql_query": "SELECT COUNT(*) AS rows_before_cleaning FROM table_name;",
        },
        {
            "question": "Explain this dataset to a new analyst.",
            "pandas_query": "df.head()",
            "sql_query": "SELECT * FROM table_name LIMIT 5;",
        },
    ]

    if quality_audit.get("findings"):
        top = quality_audit["findings"][0].get("title", "top audit issue")
        questions.insert(2, {
            "question": f"What caused the top audit finding: {top}?",
            "pandas_query": "df.head(10)",
            "sql_query": "SELECT * FROM table_name LIMIT 10;",
        })

    return questions[:8]


def suggest_queries(profiles: list[dict], context: dict | None = None) -> list[dict]:
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return _fallback_queries(profiles, context)

    columns = [
        {
            "display_name": profile.get("display_name") or profile.get("column_name"),
            "description": profile.get("description"),
        }
        for profile in profiles
    ]
    prompt = f"""
You are a data documentation expert. Based only on these column display names and descriptions, suggest 5 analytical questions a business analyst could answer using this data.

Columns:
{json.dumps(columns, indent=2, default=str)}

Return ONLY valid JSON:
[
  {{
    "question": "Plain-English question",
    "pandas_query": "One-line pandas query using df",
    "sql_query": "Standard SQL query using table_name"
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
                parsed = _parse_query_list(getattr(response, "text", ""))
                if parsed:
                    fallback = _fallback_queries(profiles, context)
                    seen = {item["question"] for item in parsed}
                    parsed.extend(item for item in fallback if item["question"] not in seen)
                    return parsed[:8]
            except Exception:
                continue
    except Exception:
        return _fallback_queries(profiles, context)
    return _fallback_queries(profiles, context)
