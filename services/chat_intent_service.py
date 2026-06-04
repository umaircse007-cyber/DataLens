import re
from enum import Enum


class ChatIntent(str, Enum):
    OFF_TOPIC = "off_topic"
    BIAS_FAIRNESS = "bias_fairness"
    COLUMN_LOOKUP = "column_lookup"
    PII_GOVERNANCE = "pii_governance"
    QUALITY_ISSUES = "quality_issues"
    TRUST_HEALTH = "trust_health"
    RELATIONSHIPS = "relationships"
    PREDICTION_TARGETS = "prediction_targets"
    EXPLAIN_AUDIENCE = "explain_audience"
    DATASET_OVERVIEW = "dataset_overview"
    ML_READINESS = "ml_readiness"
    GENERAL = "general"


OFF_TOPIC_PATTERNS = [
    r"\bweather\b",
    r"\bforecast\b",
    r"\bnews\b",
    r"\bsports?\b",
    r"\brecipe\b",
    r"\bjoke\b",
    r"\bwho won\b",
    r"\bstock price\b",
    r"\bcrypto\b",
    r"\bbitcoin\b",
    r"\bwrite (me )?a poem\b",
    r"\btranslate .+ to (french|spanish|german)\b",
    r"\bcapital of\b",
    r"\bpresident of\b",
]


def _mentioned_columns(question: str, column_names: list[str]) -> list[str]:
    lower = question.lower()
    matches = []
    for name in column_names:
        key = name.lower()
        if key and (key in lower or key.replace("_", " ") in lower):
            matches.append(name)
    return matches


def classify_intent(question: str, column_names: list[str] | None = None) -> ChatIntent:
    text = (question or "").strip().lower()
    if not text:
        return ChatIntent.GENERAL

    if any(re.search(pattern, text) for pattern in OFF_TOPIC_PATTERNS):
        return ChatIntent.OFF_TOPIC

    columns = column_names or []

    if any(token in text for token in (
        "bias", "biased", "unfair", "discriminat", "disparate", "equitable", "fairness risk",
        "protected attribute", "proxy variable", "demographic parity",
    )):
        return ChatIntent.BIAS_FAIRNESS

    if _mentioned_columns(question, columns):
        return ChatIntent.COLUMN_LOOKUP

    if any(token in text for token in ("pii", "personal data", "personal information", "gdpr", "privacy")):
        return ChatIntent.PII_GOVERNANCE

    if any(token in text for token in ("governance", "compliance", "eu ai", "retention")):
        return ChatIntent.PII_GOVERNANCE

    if any(token in text for token in (
        "quality", "problem", "issue", "wrong", "invalid", "duplicate", "outlier", "error", "clean",
        "biggest", "worst", "trustworthy",
    )):
        if "trust" in text and "quality" not in text and "issue" not in text:
            return ChatIntent.TRUST_HEALTH
        return ChatIntent.QUALITY_ISSUES

    if any(token in text for token in ("trust score", "trust", "health score", "completeness", "consistency", "validity")):
        return ChatIntent.TRUST_HEALTH

    if any(token in text for token in ("relationship", "foreign key", "primary key", "join", "correlat", "redundant")):
        return ChatIntent.RELATIONSHIPS

    if any(token in text for token in ("predict", "target", "machine learning", "model feature", "label column", "outcome")):
        return ChatIntent.PREDICTION_TARGETS

    if any(token in text for token in ("readiness", "ai ready", "analytics ready", "ml ready", "machine learning ready", "grade")):
        return ChatIntent.ML_READINESS

    if any(token in text for token in ("intern", "analyst", "executive", "business user", "non-technical", "stakeholder")):
        return ChatIntent.EXPLAIN_AUDIENCE

    if any(token in text for token in ("explain dataset", "what is this dataset", "describe dataset", "overview", "purpose", "about this data")):
        return ChatIntent.DATASET_OVERVIEW

    if any(token in text for token in ("what does", "what is", "meaning of", "define", "tell me about")):
        return ChatIntent.COLUMN_LOOKUP

    return ChatIntent.GENERAL
