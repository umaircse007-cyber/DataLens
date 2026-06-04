import re
from typing import Any


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(token) > 1}


def build_chat_index(result: dict) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    metadata = result.get("metadata") or {}
    chunks.append({
        "id": "metadata",
        "type": "metadata",
        "title": "Dataset metadata",
        "text": (
            f"File {metadata.get('filename')} has {metadata.get('row_count')} rows and "
            f"{metadata.get('column_count')} columns."
        ),
        "tokens": _tokens(str(metadata)),
    })

    dataset = result.get("dataset") or {}
    dtype = dataset.get("dataset_type") or {}
    entity = dataset.get("main_entity") or {}
    chunks.append({
        "id": "dataset_understanding",
        "type": "dataset",
        "title": "Dataset understanding",
        "text": " ".join([
            f"Dataset type: {dtype.get('label')} (confidence {dtype.get('confidence')}).",
            f"Main entity: {entity.get('label')} (confidence {entity.get('confidence')}).",
            dataset.get("business_purpose", ""),
            "Use cases: " + ", ".join(dataset.get("use_cases") or []),
            "Evidence columns: " + ", ".join(dtype.get("evidence") or []),
        ]),
        "tokens": _tokens(" ".join([str(dtype.get("label", "")), str(entity.get("label", "")), dataset.get("business_purpose", "")])),
    })

    story = result.get("story") or {}
    for key, label in (
        ("executive_summary", "Executive summary"),
        ("business_summary", "Business summary"),
        ("technical_summary", "Technical summary"),
    ):
        if story.get(key):
            chunks.append({
                "id": f"story_{key}",
                "type": "story",
                "title": label,
                "text": story[key],
                "tokens": _tokens(story[key]),
            })

    health = result.get("health") or {}
    for key, value in health.items():
        if isinstance(value, dict) and "score" in value:
            chunks.append({
                "id": f"health_{key}",
                "type": "health",
                "title": key.replace("_", " ").title(),
                "text": f"{key} score {value.get('score')}/100. {value.get('reasoning', '')}",
                "tokens": _tokens(f"{key} {value.get('reasoning', '')}"),
            })

    governance = result.get("governance") or {}
    chunks.append({
        "id": "governance",
        "type": "governance",
        "title": "Governance and compliance",
        "text": " ".join([
            f"Risk level: {governance.get('risk_level')}.",
            "PII fields: " + ", ".join(governance.get("detected_pii") or []),
            "Recommendations: " + " ".join(governance.get("recommendations") or []),
            "GDPR: " + " ".join(governance.get("gdpr_notes") or []),
        ]),
        "tokens": _tokens("governance pii gdpr eu " + " ".join(governance.get("detected_pii") or [])),
    })

    for entry in result.get("column_dictionary") or result.get("profiles") or []:
        name = str(entry.get("column_name", ""))
        semantic = (entry.get("semantic_type") or {}).get("label", "")
        sensitivity = entry.get("sensitivity") or {}
        text = " ".join([
            f"Column {name}.",
            f"Display name {entry.get('display_name', name)}.",
            f"Technical type {entry.get('technical_type') or entry.get('dtype')}.",
            f"Semantic type {semantic}.",
            f"Business description: {entry.get('business_description') or entry.get('description', '')}.",
            f"Null rate {entry.get('null_pct')}%. Unique {entry.get('unique_count')}.",
            f"Sensitivity {sensitivity.get('level')}. PII {sensitivity.get('pii')}.",
            "Quality notes: " + "; ".join(entry.get("quality_notes") or []),
        ])
        flag = entry.get("fairness_flag") or {}
        if flag:
            text += f" Fairness: {flag.get('reason', '')} EU: {flag.get('eu_ai_act_article', '')}."
        chunks.append({
            "id": f"column:{name}",
            "type": "column",
            "title": name,
            "text": text,
            "tokens": _tokens(f"{name} {semantic} {text}"),
            "column_name": name,
        })

    audit = result.get("quality_audit") or {}
    for finding in audit.get("findings") or []:
        chunks.append({
            "id": finding.get("id", f"finding_{len(chunks)}"),
            "type": "finding",
            "title": finding.get("title", "Finding"),
            "text": " ".join([
                f"{finding.get('severity')} {finding.get('category')}: {finding.get('title')}.",
                finding.get("summary", ""),
                "Columns: " + ", ".join(finding.get("source_columns") or []),
                finding.get("recommendation", ""),
            ]),
            "tokens": _tokens(
                f"{finding.get('title')} {finding.get('summary')} "
                f"{' '.join(finding.get('source_columns') or [])}"
            ),
        })

    for outlier in (audit.get("outliers") or [])[:25]:
        chunks.append({
            "id": f"outlier:{outlier.get('column')}:{outlier.get('row_index')}",
            "type": "outlier",
            "title": f"Outlier in {outlier.get('column')}",
            "text": outlier.get("explanation", ""),
            "tokens": _tokens(f"{outlier.get('column')} outlier {outlier.get('explanation', '')}"),
        })

    for rel in (result.get("relationships") or [])[:30]:
        chunks.append({
            "id": f"rel:{rel.get('col_a')}:{rel.get('col_b')}",
            "type": "relationship",
            "title": f"{rel.get('col_a')} ↔ {rel.get('col_b')}",
            "text": f"{rel.get('type')}: {rel.get('note')} correlation {rel.get('correlation')}.",
            "tokens": _tokens(f"{rel.get('col_a')} {rel.get('col_b')} {rel.get('type')}"),
        })

    readiness = result.get("readiness") or {}
    chunks.append({
        "id": "readiness",
        "type": "readiness",
        "title": "AI & Analytics Readiness",
        "text": (
            f"AI & Analytics Readiness score {readiness.get('score')}/100 grade {readiness.get('grade')}. "
            f"{readiness.get('summary', '')}"
        ),
        "tokens": _tokens("readiness ai analytics model " + readiness.get("summary", "")),
    })

    fairness_lines = []
    for entry in result.get("column_dictionary") or result.get("profiles") or []:
        flag = entry.get("fairness_flag") or {}
        if not flag:
            continue
        groq = flag.get("groq_verification") or {}
        fairness_lines.append(
            f"Column {entry.get('column_name')}: sensitive={flag.get('is_sensitive')} proxy={flag.get('is_proxy')}. "
            f"Reason: {flag.get('reason', '')}. EU: {flag.get('eu_ai_act_article', '')}. "
            f"Groq verdict: {groq.get('verdict', 'n/a')} — {groq.get('reason', '')}."
        )
    if fairness_lines:
        chunks.append({
            "id": "fairness_flags",
            "type": "fairness",
            "title": "Fairness and bias review",
            "text": " ".join(fairness_lines),
            "tokens": _tokens("bias fairness discrimination proxy sensitive groq " + " ".join(fairness_lines)),
        })

    return chunks


def retrieve_chunks(question: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
    question_tokens = _tokens(question)
    if not question_tokens:
        return chunks[:top_k]

    scored = []
    for chunk in chunks:
        chunk_tokens = chunk.get("tokens") or _tokens(chunk.get("text", ""))
        overlap = len(question_tokens & chunk_tokens)
        column_name = (chunk.get("column_name") or "").lower()
        name_boost = 3 if column_name and column_name in question.lower() else 0
        title_boost = 2 if chunk.get("title", "").lower() in question.lower() else 0
        score = overlap + name_boost + title_boost
        if score > 0 or chunk.get("type") in {"metadata", "dataset"}:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return chunks[:top_k]
    return [chunk for _, chunk in scored[:top_k]]
