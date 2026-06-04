import re
from itertools import combinations
from typing import Any

import pandas as pd


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _pk_candidates(df: pd.DataFrame, schema_map: dict[str, dict]) -> list[str]:
    candidates = []
    row_count = len(df)
    for column in df.columns:
        name = str(column)
        semantic = (schema_map.get(name) or {}).get("label", "")
        series = df[column].dropna()
        if semantic == "Primary Key Candidate":
            candidates.append(name)
            continue
        if row_count and len(series) == row_count and series.nunique() == row_count:
            if _norm(name).endswith("_id") or _norm(name) == "id":
                candidates.append(name)
    return candidates


def _overlap_score(left: pd.Series, right: pd.Series) -> float:
    left_values = set(left.dropna().astype(str).str.strip())
    right_values = set(right.dropna().astype(str).str.strip())
    if not left_values or not right_values:
        return 0.0
    return len(left_values & right_values) / min(len(left_values), len(right_values))


def build_relationship_graph(
    table_name: str,
    df: pd.DataFrame,
    schema_map: dict[str, dict],
    intra_relationships: list[dict],
) -> dict[str, Any]:
    pk_columns = _pk_candidates(df, schema_map)
    nodes = [{"id": f"{table_name}.{column}", "label": str(column), "table": table_name, "role": "PK" if column in pk_columns else "column"} for column in df.columns]
    edges = []

    for relationship in intra_relationships:
        edges.append({
            "source": f"{table_name}.{relationship.get('col_a')}",
            "target": f"{table_name}.{relationship.get('col_b')}",
            "type": relationship.get("type", "correlation"),
            "confidence": min(0.99, abs(float(relationship.get("correlation") or 0.8))),
            "note": relationship.get("note", ""),
        })

    for col_a, col_b in combinations(df.columns, 2):
        name_a, name_b = _norm(str(col_a)), _norm(str(col_b))
        if name_a.endswith("_id") and name_b.replace("_id", "") in name_a:
            edges.append({
                "source": f"{table_name}.{col_a}",
                "target": f"{table_name}.{col_b}",
                "type": "derived-key-hint",
                "confidence": 0.7,
                "note": f"{col_a} naming suggests relationship to {col_b}.",
            })

    return {"nodes": nodes, "edges": edges, "primary_keys": pk_columns}


def discover_cross_table_relationships(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(tables) < 2:
        return []

    relationships = []
    for left_table, right_table in combinations(tables, 2):
        left_name = left_table["name"]
        right_name = right_table["name"]
        left_df = left_table["df"]
        right_df = right_table["df"]

        for left_col in left_df.columns:
            for right_col in right_df.columns:
                if _norm(str(left_col)) != _norm(str(right_col)) and not (
                    _norm(str(left_col)).endswith("_id") and _norm(str(right_col)).endswith("_id")
                ):
                    continue
                overlap = _overlap_score(left_df[left_col], right_df[right_col])
                if overlap < 0.6:
                    continue
                left_unique = left_df[left_col].dropna().nunique()
                right_unique = right_df[right_col].dropna().nunique()
                rel_type = "One-to-Many" if left_unique >= right_unique else "Many-to-One"
                relationships.append({
                    "left_table": left_name,
                    "right_table": right_name,
                    "left_column": str(left_col),
                    "right_column": str(right_col),
                    "relationship_type": rel_type,
                    "confidence": round(min(0.98, 0.55 + overlap * 0.4), 2),
                    "evidence": [f"Value overlap {round(overlap * 100, 1)}% on matching column names"],
                })
    return relationships
