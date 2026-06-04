import json
import os
import re
from typing import Any

from services.chat_index_service import build_chat_index, retrieve_chunks
from services.chat_intent_service import ChatIntent, classify_intent
from services.groq_client import groq_chat, is_groq_configured
from services.groq_keys import ensure_env_loaded

REFUSAL_MESSAGE = (
    "I can only answer questions about this uploaded dataset and DataLens analysis results. "
    "I cannot help with general knowledge such as weather, news, or topics outside this session."
)

# Intents that must use LLM analysis (not instant rule templates).
LLM_FIRST_INTENTS = {
    ChatIntent.BIAS_FAIRNESS,
    ChatIntent.GENERAL,
    ChatIntent.EXPLAIN_AUDIENCE,
    ChatIntent.DATASET_OVERVIEW,
}

# Simple factual intents can use fast structured answers.
FAST_INTENTS = {
    ChatIntent.COLUMN_LOOKUP,
    ChatIntent.PII_GOVERNANCE,
    ChatIntent.QUALITY_ISSUES,
    ChatIntent.TRUST_HEALTH,
    ChatIntent.RELATIONSHIPS,
    ChatIntent.PREDICTION_TARGETS,
    ChatIntent.ML_READINESS,
}


def ai_provider_status() -> dict[str, Any]:
    ensure_env_loaded()
    gemini_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    return {
        "gemini_configured": bool(gemini_key) and not gemini_key.lower().startswith("your_"),
        "groq_configured": is_groq_configured(),
    }


def _column_names(result: dict) -> list[str]:
    dictionary = result.get("column_dictionary") or result.get("profiles") or []
    return [str(item.get("column_name", "")) for item in dictionary if item.get("column_name")]


def _dictionary_map(result: dict) -> dict[str, dict]:
    dictionary = result.get("column_dictionary") or result.get("profiles") or []
    return {str(item.get("column_name", "")): item for item in dictionary if item.get("column_name")}


def _format_retrieved_evidence(chunks: list[dict]) -> str:
    return "\n".join(
        f"[{chunk.get('type', 'info').upper()}] {chunk.get('title', 'Evidence')}: {chunk.get('text', '')}"
        for chunk in chunks
    )


def build_evidence_bundle(result: dict, question: str, intent: ChatIntent) -> str:
    index = result.get("chat_index") or build_chat_index(result)
    top_k = 16 if intent == ChatIntent.BIAS_FAIRNESS else 12
    chunks = retrieve_chunks(question, index, top_k=top_k)

    if intent == ChatIntent.BIAS_FAIRNESS:
        chunks = [chunk for chunk in chunks if chunk.get("type") in {"fairness", "governance", "column", "finding", "health"}] or chunks

    sections = [_format_retrieved_evidence(chunks)]

    if intent == ChatIntent.BIAS_FAIRNESS:
        sections.append(_build_bias_evidence_block(result))

    return "\n\n".join(section for section in sections if section)


def _build_bias_evidence_block(result: dict) -> str:
    lines = ["[FAIRNESS ANALYSIS EVIDENCE]"]
    flagged = []
    for entry in result.get("column_dictionary") or result.get("profiles") or []:
        flag = entry.get("fairness_flag") or {}
        if not flag:
            continue
        groq = flag.get("groq_verification") or {}
        flagged.append(entry.get("column_name"))
        lines.append(
            f"- {entry.get('column_name')}: sensitive={flag.get('is_sensitive')}, proxy={flag.get('is_proxy')}. "
            f"Reason: {flag.get('reason', '')}. "
            f"EU mapping: {flag.get('eu_ai_act_article', 'n/a')}. "
            f"Groq verification: {groq.get('verdict', 'n/a')} ({groq.get('reason', '')})."
        )

    gov = result.get("governance") or {}
    health = result.get("health") or {}
    readiness = result.get("readiness") or {}
    lines.extend([
        f"Flagged fairness columns ({len(flagged)}): {', '.join(flagged) if flagged else 'none'}.",
        f"Governance risk: {gov.get('risk_level', 'unknown')}. PII fields: {', '.join(gov.get('detected_pii') or []) or 'none'}.",
        f"Governance score: {(health.get('governance') or {}).get('score', 'n/a')}/100.",
        f"AI & Analytics Readiness: {readiness.get('score', 'n/a')}/100 — sensitive-column review may apply.",
        "Note: Schema-level review cannot prove outcome bias without a labeled decision column and group statistics.",
    ])
    return "\n".join(lines)


def _answer_column_lookup(question: str, result: dict) -> str | None:
    columns = _dictionary_map(result)
    lower = question.lower()
    matched = [name for name in columns if name.lower() in lower or name.lower().replace("_", " ") in lower]
    if not matched:
        tokens = re.findall(r"[a-z0-9_]+", lower)
        matched = [name for name in columns if any(token in name.lower() for token in tokens if len(token) > 3)]
    if not matched:
        return None

    parts = []
    for name in matched[:3]:
        entry = columns[name]
        semantic = entry.get("semantic_type") or {}
        sensitivity = entry.get("sensitivity") or {}
        parts.append(
            f"**{entry.get('display_name') or name}**\n"
            f"- Semantic type: {semantic.get('label', 'Unknown')} ({int(float(semantic.get('confidence', 0)) * 100)}%)\n"
            f"- Technical type: {entry.get('technical_type') or entry.get('dtype')}\n"
            f"- Description: {entry.get('business_description') or entry.get('description') or 'Not documented.'}\n"
            f"- Sensitivity: {sensitivity.get('level', 'Low')}"
            + (" (PII)" if sensitivity.get("pii") else "")
        )
    return "\n\n".join(parts)


def _answer_pii_governance(result: dict) -> str:
    gov = result.get("governance") or {}
    pii = gov.get("detected_pii") or []
    lines = [
        f"**Governance risk level:** {gov.get('risk_level', 'Unknown')}",
        f"**PII fields:** {', '.join(pii) if pii else 'None flagged.'}",
    ]
    return "\n".join(lines)


def _answer_quality_issues(result: dict, question: str) -> str:
    audit = result.get("quality_audit") or {}
    findings = audit.get("findings") or []
    limit = 5 if any(token in question.lower() for token in ("5", "five", "biggest", "top")) else 8
    if not findings:
        return "No material quality issues were detected."
    lines = [f"**Audit score:** {audit.get('score', 'n/a')}/100", ""]
    for index, finding in enumerate(findings[:limit], start=1):
        lines.append(f"{index}. **{finding.get('severity')} — {finding.get('title')}**: {finding.get('summary')}")
    return "\n".join(lines)


def _answer_trust_health(result: dict) -> str:
    health = result.get("health") or {}
    trust = health.get("overall_trust") or {}
    lines = [f"**Trust score: {trust.get('score', 'n/a')}/100**", trust.get("reasoning", "")]
    for key in ("completeness", "consistency", "validity", "uniqueness", "governance"):
        dim = health.get(key) or {}
        if dim:
            lines.append(f"- {key.title()}: {dim.get('score')}/100")
    return "\n".join(lines)


def _answer_relationships(result: dict) -> str:
    lines = []
    for rel in (result.get("relationships") or [])[:8]:
        lines.append(f"- {rel.get('col_a')} ↔ {rel.get('col_b')}: {rel.get('note')}")
    return "\n".join(lines) or "No strong relationships detected."


def _answer_prediction_targets(result: dict) -> str:
    good, avoid = [], []
    for entry in result.get("column_dictionary") or []:
        name = entry.get("column_name", "")
        if (entry.get("fairness_flag") or {}).get("is_sensitive"):
            avoid.append(name)
        elif (entry.get("semantic_type") or {}).get("label") not in {"Primary Key Candidate", "Identifier"}:
            good.append(name)
    return f"**Good targets:** {', '.join(good[:8]) or 'none'}\n**Avoid:** {', '.join(avoid[:8]) or 'none'}"


def _answer_audience(result: dict, question: str) -> str:
    story = result.get("story") or {}
    lower = question.lower()
    if "intern" in lower:
        return story.get("business_summary") or story.get("executive_summary", "")
    if "analyst" in lower:
        return story.get("technical_summary") or story.get("business_summary", "")
    return story.get("executive_summary", "")


def _answer_dataset_overview(result: dict) -> str:
    return (result.get("story") or {}).get("executive_summary", "")


def _answer_ml_readiness(result: dict) -> str:
    r = result.get("readiness") or {}
    factors = "\n".join(
        f"- **{item.get('title')}**: {item.get('readiness_impact')} impact"
        for item in (r.get("factors") or [])[:6]
    )
    return f"**AI & Analytics Readiness: {r.get('score')}/100 (grade {r.get('grade')})**\n{r.get('summary', '')}\n{factors}"


def _rule_based_answer(intent: ChatIntent, question: str, result: dict) -> str | None:
    handlers = {
        ChatIntent.COLUMN_LOOKUP: lambda: _answer_column_lookup(question, result),
        ChatIntent.PII_GOVERNANCE: lambda: _answer_pii_governance(result),
        ChatIntent.QUALITY_ISSUES: lambda: _answer_quality_issues(result, question),
        ChatIntent.TRUST_HEALTH: lambda: _answer_trust_health(result),
        ChatIntent.RELATIONSHIPS: lambda: _answer_relationships(result),
        ChatIntent.PREDICTION_TARGETS: lambda: _answer_prediction_targets(result),
        ChatIntent.EXPLAIN_AUDIENCE: lambda: _answer_audience(result, question),
        ChatIntent.DATASET_OVERVIEW: lambda: _answer_dataset_overview(result),
        ChatIntent.ML_READINESS: lambda: _answer_ml_readiness(result),
    }
    handler = handlers.get(intent)
    return handler() if handler else None


def _call_gemini(question: str, evidence: str, intent: str) -> tuple[str | None, str | None]:
    ensure_env_loaded()
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key or api_key.lower().startswith("your_"):
        return None, "GEMINI_API_KEY is not set in .env"

    bias_extra = ""
    if intent == ChatIntent.BIAS_FAIRNESS.value:
        bias_extra = (
            " The user is asking about bias or fairness. Give a direct yes/no/uncertain assessment, "
            "cite flagged columns and Groq verification verdicts, explain limits of schema-only review, "
            "and list concrete next steps (e.g. disparate impact test if outcome column exists)."
        )

    system = (
        "You are DataLens AI Data Steward. Answer ONLY the user's specific question using the evidence. "
        "Do NOT paste a generic audit summary unless asked. Use markdown with a clear verdict section."
        + bias_extra
    )
    user = f"Question: {question}\n\nEvidence:\n{evidence}"

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        model_names = [
            (os.environ.get("GEMINI_MODEL") or "").replace("models/", "").strip(),
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ]
        last_error = "Gemini did not return a response."
        for model_name in [name for name in model_names if name]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=f"{system}\n\n{user}",
                )
                text = (getattr(response, "text", "") or "").strip()
                if text:
                    return text, None
            except Exception as exc:
                last_error = str(exc)
        return None, last_error
    except Exception as exc:
        return None, str(exc)


def _verify_answer_with_groq(question: str, draft: str, evidence: str, intent: str) -> tuple[str | None, str | None]:
    ensure_env_loaded()
    if not is_groq_configured():
        return None, "Groq API key not configured"

    prompt = f"""
You are Groq, the independent verification layer for DataLens chat answers.

The user asked:
{question}

Draft answer to verify:
{draft}

Evidence (only facts you may use):
{evidence}

Tasks:
1. Check whether the draft directly answers the user's question (especially for bias/fairness questions).
2. Remove generic filler that does not address the question.
3. Correct any claim not supported by the evidence.
4. Return the improved final answer in professional markdown.
5. For bias questions, state clearly: evidence of risk vs proof of bias, and what is unknown.

Return ONLY the final answer text, no preamble.
"""
    return groq_chat(
        [
            {"role": "system", "content": "You verify and improve DataLens dataset answers. Be precise and evidence-based."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )


def _compose_bias_fallback(result: dict) -> str:
    flagged = [
        entry for entry in (result.get("column_dictionary") or [])
        if entry.get("fairness_flag")
    ]
    if not flagged:
        return (
            "**Bias assessment (schema review):** No sensitive or proxy columns were flagged in this dataset. "
            "That does not prove the data is unbiased for modeling — only that obvious fairness risks were not detected."
        )

    lines = [
        "**Bias assessment (based on DataLens fairness scan):**",
        f"{len(flagged)} column(s) raise fairness or proxy concerns:",
    ]
    for entry in flagged[:8]:
        flag = entry["fairness_flag"]
        groq = flag.get("groq_verification") or {}
        lines.append(
            f"- **{entry.get('column_name')}** ({'sensitive' if flag.get('is_sensitive') else 'proxy'}): "
            f"{flag.get('reason', '')} Groq: {groq.get('verdict', 'n/a')}."
        )
    lines.append(
        "\n**Conclusion:** The dataset shows **fairness risk signals** that require review before ML or automated decisions. "
        "This is not proof of discriminatory outcomes without testing against an outcome and protected groups."
    )
    return "\n".join(lines)


def answer_audit_question(question: str, result: dict) -> dict:
    ensure_env_loaded()
    question = (question or "").strip()
    column_names = _column_names(result)
    intent = classify_intent(question, column_names)
    status = ai_provider_status()

    if intent == ChatIntent.OFF_TOPIC:
        return {
            "answer": REFUSAL_MESSAGE,
            "in_scope": False,
            "intent": intent.value,
            "source": "policy",
            "ai_status": status,
            "groq_verified": False,
            "models": {"gemini": False, "groq": False},
        }

    evidence = build_evidence_bundle(result, question, intent)
    rule_answer = _rule_based_answer(intent, question, result)
    gemini_answer, gemini_error = None, None
    groq_answer, groq_error = None, None
    groq_verified = False

    use_llm_pipeline = intent in LLM_FIRST_INTENTS

    if use_llm_pipeline:
        gemini_answer, gemini_error = _call_gemini(question, evidence, intent.value)
        if intent == ChatIntent.BIAS_FAIRNESS:
            draft = gemini_answer or rule_answer or _compose_bias_fallback(result)
        else:
            draft = gemini_answer or rule_answer or ""

        if is_groq_configured() and draft:
            groq_answer, groq_error = _verify_answer_with_groq(question, draft, evidence, intent.value)
            if groq_answer:
                groq_verified = True
                final = groq_answer
                source = "groq_verified"
            else:
                final = draft
                source = "gemini" if gemini_answer else ("rules" if rule_answer else "retrieval")
        elif gemini_answer:
            final = gemini_answer
            source = "gemini"
        elif intent == ChatIntent.BIAS_FAIRNESS:
            final = _compose_bias_fallback(result)
            source = "rules"
        elif rule_answer:
            final = rule_answer
            source = "rules"
        else:
            index = result.get("chat_index") or build_chat_index(result)
            chunks = retrieve_chunks(question, index, top_k=8)
            final = _format_retrieved_evidence(chunks[:5])
            source = "retrieval"
    else:
        if intent in FAST_INTENTS and rule_answer:
            final = rule_answer
            source = "rules"
        else:
            gemini_answer, gemini_error = _call_gemini(question, evidence, intent.value)
            if is_groq_configured() and gemini_answer:
                groq_answer, groq_error = _verify_answer_with_groq(question, gemini_answer, evidence, intent.value)
                if groq_answer:
                    groq_verified = True
                    final = groq_answer
                    source = "groq_verified"
                else:
                    final = gemini_answer
                    source = "gemini"
            elif gemini_answer:
                final = gemini_answer
                source = "gemini"
            elif rule_answer:
                final = rule_answer
                source = "rules"
            else:
                final = "I could not find enough context to answer that question."
                source = "retrieval"

    response = {
        "answer": final,
        "in_scope": True,
        "intent": intent.value,
        "source": source,
        "groq_verified": groq_verified,
        "ai_status": status,
        "models": {
            "gemini": bool(gemini_answer),
            "groq": bool(groq_answer) or groq_verified,
        },
    }
    if gemini_error and not gemini_answer:
        response["gemini_error"] = gemini_error
    if groq_error and not groq_answer and is_groq_configured():
        response["groq_error"] = groq_error
    return response


def is_in_scope(question: str) -> bool:
    return classify_intent(question) != ChatIntent.OFF_TOPIC
