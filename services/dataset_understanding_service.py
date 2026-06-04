import re
from typing import Any

import pandas as pd

DATASET_SIGNATURES = {
    "HR Dataset": ("salary", "employee", "department", "hire", "compensation", "gender", "dob"),
    "Customer Dataset": ("customer", "email", "phone", "address", "loyalty", "account"),
    "Financial Dataset": ("transaction", "amount", "revenue", "invoice", "payment", "currency", "balance"),
    "Healthcare Dataset": ("patient", "diagnosis", "provider", "medical", "prescription", "icd"),
    "E-commerce Dataset": ("order", "product", "cart", "sku", "shipping", "checkout"),
    "Educational Dataset": ("student", "course", "grade", "enrollment", "campus", "tuition"),
}

ENTITY_SIGNATURES = {
    "Employee": ("employee", "salary", "department", "hire"),
    "Customer": ("customer", "email", "account"),
    "Patient": ("patient", "diagnosis", "medical"),
    "Order": ("order", "product", "sku"),
    "Transaction": ("transaction", "payment", "amount"),
    "Student": ("student", "course", "grade"),
}


def _norm_columns(columns: list[str]) -> set[str]:
    return {re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_") for column in columns}


def _score_signature(columns: set[str], hints: tuple[str, ...]) -> tuple[int, list[str]]:
    matched = [hint for hint in hints if any(hint == col or hint in col for col in columns)]
    return len(matched), matched


def _evidence_label(hint: str) -> str:
    labels = {
        "salary": "salary -> compensation indicator",
        "compensation": "compensation -> workforce pay indicator",
        "department": "department -> organizational indicator",
        "gender": "gender -> workforce demographic indicator",
        "dob": "dob -> employee demographic indicator",
        "date_of_birth": "date_of_birth -> employee demographic indicator",
        "customer": "customer -> customer identity indicator",
        "email": "email -> contact information indicator",
        "phone": "phone -> contact information indicator",
        "currency": "currency -> financial geography indicator",
        "amount": "amount -> financial measure indicator",
        "student": "student -> education entity indicator",
        "grade": "grade -> education performance indicator",
    }
    return labels.get(hint, f"{hint} -> matching schema indicator")


def _rank_matches(options: dict[str, tuple[str, ...]], columns: set[str]) -> list[dict[str, Any]]:
    ranked = []
    for label, hints in options.items():
        score, evidence = _score_signature(columns, hints)
        ranked.append({
            "label": label,
            "raw_score": score,
            "evidence": [_evidence_label(item) for item in evidence[:8]],
        })
    ranked.sort(key=lambda item: item["raw_score"], reverse=True)
    total = sum(item["raw_score"] for item in ranked) or 1
    for item in ranked:
        item["confidence"] = round((item["raw_score"] / total) if item["raw_score"] else 0.0, 2)
    return ranked


def _best_match(options: dict[str, tuple[str, ...]], columns: set[str]) -> dict[str, Any]:
    ranked = _rank_matches(options, columns)
    best = ranked[0] if ranked else {"label": "General Business Dataset", "raw_score": 0, "evidence": [], "confidence": 0}
    best_score = int(best.get("raw_score") or 0)
    confidence = min(0.98, 0.45 + best_score * 0.12) if best_score else 0.5
    alternatives = [
        {
            "label": item["label"],
            "confidence": item["confidence"],
            "evidence": item["evidence"],
        }
        for item in ranked[1:4]
    ]
    reasoning = (
        f"The schema contains {best_score} strong indicator(s) for {best['label']}."
        if best_score
        else "No single domain signature dominated, so the dataset is treated as a general business dataset."
    )
    return {
        "label": best["label"] if best_score else "General Business Dataset",
        "confidence": round(confidence, 2),
        "confidence_score": int(round(confidence * 100)),
        "evidence": best["evidence"][:8],
        "reasoning": reasoning,
        "alternative_matches": alternatives,
    }


def understand_dataset(df: pd.DataFrame, profiles: list[dict]) -> dict[str, Any]:
    columns = _norm_columns([profile.get("column_name", "") for profile in profiles])
    dataset_type = _best_match(DATASET_SIGNATURES, columns)
    main_entity = _best_match(ENTITY_SIGNATURES, columns)

    use_cases = []
    if dataset_type["label"] == "HR Dataset":
        use_cases = ["Workforce planning", "Salary benchmarking", "Diversity analysis", "HR reporting"]
        purpose = (
            "This dataset appears to contain workforce demographic and compensation information "
            "used for HR analytics and organizational reporting."
        )
    elif dataset_type["label"] == "Customer Dataset":
        use_cases = ["Customer segmentation", "Retention analysis", "Contact data quality review"]
        purpose = "This dataset appears to describe customers or accounts for CRM and marketing analytics."
    elif dataset_type["label"] == "Financial Dataset":
        use_cases = ["Revenue reporting", "Payment reconciliation", "Risk monitoring"]
        purpose = "This dataset appears to support financial reporting, billing, or transaction analysis."
    else:
        use_cases = ["Exploratory analysis", "Operational reporting", "Data quality remediation"]
        purpose = (
            "This dataset appears to support operational or analytical reporting, "
            "but no single domain signature was dominant."
        )

    return {
        "dataset_type": dataset_type,
        "main_entity": main_entity,
        "business_purpose": purpose,
        "use_cases": use_cases,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
    }
