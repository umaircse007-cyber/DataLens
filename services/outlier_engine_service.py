from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest


def _severity_from_ratio(ratio: float) -> str:
    if ratio >= 0.05:
        return "Critical"
    if ratio >= 0.02:
        return "High"
    if ratio >= 0.01:
        return "Medium"
    return "Low"


def _outlier_record(
    *,
    column: str,
    row_index: Any,
    value: Any,
    method: str,
    severity: str,
    explanation: str,
) -> dict[str, Any]:
    return {
        "column": column,
        "row_index": int(row_index) if str(row_index).isdigit() else row_index,
        "value": value,
        "method": method,
        "severity": severity,
        "explanation": explanation,
    }


def detect_outliers(df: pd.DataFrame, max_per_column: int = 12) -> list[dict[str, Any]]:
    outliers: list[dict[str, Any]] = []
    row_count = len(df)

    for column in df.select_dtypes(include="number").columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(series) < 4:
            continue

        values = series.astype(float)
        mean = values.mean()
        std = values.std()
        if std:
            z_scores = (values - mean).abs() / std
            z_outliers = values[z_scores > 3]
            for idx, value in z_outliers.head(max_per_column).items():
                pct = float((values <= value).mean() * 100)
                outliers.append(_outlier_record(
                    column=str(column),
                    row_index=idx,
                    value=float(value),
                    method="Z-score",
                    severity=_severity_from_ratio(len(z_outliers) / row_count),
                    explanation=(
                        f"{column} value {value:,.4g} is more than 3 standard deviations from the mean "
                        f"and exceeds about {pct:.1f}% of records."
                    ),
                ))

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr:
            upper = q3 + 1.5 * iqr
            lower = q1 - 1.5 * iqr
            iqr_outliers = values[(values > upper) | (values < lower)]
            for idx, value in iqr_outliers.head(max_per_column).items():
                if any(item["column"] == str(column) and item["row_index"] == idx for item in outliers):
                    continue
                direction = "high" if value > upper else "low"
                typical = values[(values <= upper) & (values >= lower)]
                typical_bound = typical.max() if direction == "high" else typical.min()
                outliers.append(_outlier_record(
                    column=str(column),
                    row_index=idx,
                    value=float(value),
                    method="IQR",
                    severity=_severity_from_ratio(len(iqr_outliers) / row_count),
                    explanation=(
                        f"{column} value {value:,.4g} is an extreme {direction} outlier; "
                        f"typical values are near {float(typical_bound):,.4g}."
                    ),
                ))

        if len(values) >= 20:
            matrix = values.to_numpy().reshape(-1, 1)
            model = IsolationForest(contamination=min(0.1, max(0.02, 5 / len(values))), random_state=42)
            predictions = model.fit_predict(matrix)
            iso_indices = values.index[predictions == -1]
            for idx in iso_indices[:max(4, max_per_column // 3)]:
                if any(item["column"] == str(column) and item["row_index"] == idx for item in outliers):
                    continue
                value = float(values.loc[idx])
                outliers.append(_outlier_record(
                    column=str(column),
                    row_index=idx,
                    value=value,
                    method="Isolation Forest",
                    severity="Medium",
                    explanation=(
                        f"{column} value {value:,.4g} was flagged as anomalous by multivariate isolation scoring."
                    ),
                ))

    outliers.sort(key=lambda item: {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}.get(item["severity"], 9))
    return outliers[:80]
