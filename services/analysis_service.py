import asyncio
from datetime import datetime
from typing import Any

import pandas as pd

from services.ai_scan_service import scan_all_columns
from services.anomaly_service import add_anomaly_notes
from services.cross_table_relationship_service import (
    build_relationship_graph,
    discover_cross_table_relationships,
)
from services.data_service import profile_dataset
from services.dataset_story_service import build_dataset_story
from services.dataset_understanding_service import understand_dataset
from services.dictionary_generator_service import build_column_dictionary
from services.fairness_flag_service import flag_sensitive_columns
from services.governance_agent_service import build_governance
from services.health_scoring_service import build_health_scores
from services.outlier_engine_service import detect_outliers
from services.quality_audit_service import build_quality_audit
from services.query_service import suggest_queries
from services.readiness_service import ml_readiness_score
from services.relationship_service import detect_redundant_columns, detect_relationships
from services.schema_intelligence_service import build_schema_intelligence
from services.chat_index_service import build_chat_index


async def run_full_analysis(
    df: pd.DataFrame,
    filename: str,
    session_id: str,
) -> dict[str, Any]:
    profiles = profile_dataset(df)
    schema_map = build_schema_intelligence(df)
    profiles = await scan_all_columns(df, profiles)
    profiles = add_anomaly_notes(profiles, df)
    relationships = detect_relationships(df)
    redundant_columns = detect_redundant_columns(df)
    outliers = detect_outliers(df)
    quality_audit = build_quality_audit(df, profiles)
    quality_audit["outliers"] = outliers

    profiles = await asyncio.to_thread(flag_sensitive_columns, profiles, df)
    column_dictionary = build_column_dictionary(profiles, schema_map, quality_audit, outliers)
    understanding = understand_dataset(df, profiles)
    governance = build_governance(column_dictionary, profiles, quality_audit)
    health = build_health_scores(profiles, quality_audit, governance, column_dictionary)
    readiness = ml_readiness_score(df, profiles, quality_audit=quality_audit, governance=governance)
    query_suggestions = await asyncio.to_thread(
        suggest_queries,
        column_dictionary,
        {
            "dataset": understanding,
            "quality_audit": quality_audit,
            "health": health,
            "governance": governance,
            "readiness": readiness,
        },
    )
    story = build_dataset_story(
        {
            "filename": filename,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
        },
        understanding,
        health,
        quality_audit,
        column_dictionary,
    )
    relationship_graph = build_relationship_graph(
        table_name=filename,
        df=df,
        schema_map=schema_map,
        intra_relationships=relationships,
    )

    flagged_column_count = sum(1 for profile in profiles if profile.get("fairness_flag"))
    result_payload = {
        "analysis_version": 2,
        "metadata": {
            "session_id": session_id,
            "filename": filename,
            "timestamp": datetime.utcnow().isoformat(),
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "flagged_column_count": int(flagged_column_count),
        },
        "dataset": understanding,
        "story": story,
        "column_dictionary": column_dictionary,
        "schema_intelligence": schema_map,
        "profiles": profiles,
        "relationships": relationships,
        "redundant_columns": redundant_columns,
        "relationship_graph": relationship_graph,
        "cross_table_relationships": [],
        "quality_audit": quality_audit,
        "health": health,
        "governance": governance,
        "query_suggestions": query_suggestions,
        "readiness": readiness,
        "exports": {
            "pdf": f"/dictionary/export/{session_id}/pdf",
            "excel": f"/dictionary/export/{session_id}/excel",
            "json": f"/dictionary/export/{session_id}/json",
            "markdown": f"/dictionary/export/{session_id}/markdown",
            "audit_csv": f"/dictionary/export/{session_id}/audit_csv",
            "governance": f"/dictionary/export/{session_id}/governance",
        },
    }
    result_payload["chat_index"] = build_chat_index(result_payload)
    return result_payload


async def run_multi_table_analysis(
    tables: list[tuple[str, pd.DataFrame]],
    session_id: str,
    primary_filename: str,
) -> dict[str, Any]:
    primary_name, primary_df = tables[0]
    result = await run_full_analysis(primary_df, primary_filename, session_id)
    table_payloads = []
    graphs = [result["relationship_graph"]]

    for name, frame in tables:
        schema_map = build_schema_intelligence(frame)
        intra = detect_relationships(frame)
        table_payloads.append({"name": name, "df": frame, "schema_map": schema_map})
        if name != primary_name:
            graphs.append(build_relationship_graph(name, frame, schema_map, intra))

    cross = discover_cross_table_relationships(
        [{"name": item["name"], "df": item["df"]} for item in table_payloads]
    )
    result["cross_table_relationships"] = cross
    result["relationship_graphs"] = graphs
    result["tables_analysed"] = [name for name, _ in tables]
    result["chat_index"] = build_chat_index(result)
    return result
