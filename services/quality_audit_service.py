import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
URL_RE = re.compile(r"^https?://[^\s]+$", re.I)

COUNTRY_ALIASES = {
    "usa": "United States",
    "us": "United States",
    "u.s.a.": "United States",
    "u.s.a": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "america": "United States",
    "india": "India",
    "bharat": "India",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom",
}

EXPECTED_CURRENCIES = {
    "India": {"inr", "rupee", "rupees", "indian rupee"},
    "United States": {"usd", "dollar", "dollars", "us dollar", "american dollar"},
    "United Kingdom": {"gbp", "pound", "pounds", "sterling", "british pound"},
}

SEVERITY_POINTS = {
    "Critical": 12,
    "High": 9,
    "Medium": 5,
    "Low": 2,
}


def _safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _column_by_hints(df: pd.DataFrame, hints: tuple[str, ...]) -> str | None:
    for column in df.columns:
        key = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        if any(hint == key or hint in key for hint in hints):
            return str(column)
    return None


def _row_label(index: Any) -> str:
    try:
        return f"row {int(index) + 1}"
    except Exception:
        return f"row {index}"


def _finding(
    findings: list[dict],
    *,
    category: str,
    severity: str,
    title: str,
    summary: str,
    evidence: list[str] | None = None,
    recommendation: str,
    affected_rows: int = 0,
    row_count: int = 0,
    source_columns: list[str] | None = None,
    confidence: float | None = None,
) -> None:
    findings.append({
        "id": f"{category.lower().replace(' ', '_')}_{len(findings) + 1}",
        "category": category,
        "severity": severity,
        "title": title,
        "summary": summary,
        "evidence": evidence or [],
        "recommendation": recommendation,
        "affected_rows": int(affected_rows),
        "affected_pct": round((affected_rows / row_count * 100), 1) if row_count else 0,
        "source_columns": source_columns or [],
        "confidence": confidence,
    })


def _fuzzy_name_pairs(unique_names: list[str]) -> list[tuple[str, str, int]]:
    if len(unique_names) < 2:
        return []

    pairs: dict[tuple[str, str], int] = {}
    for i, left in enumerate(unique_names):
        for right in unique_names[i + 1:]:
            if left.lower() == right.lower():
                continue
            ratio = SequenceMatcher(None, left.lower(), right.lower()).ratio()
            if ratio >= 0.88:
                key = tuple(sorted((left, right)))
                pairs[key] = max(pairs.get(key, 0), round(ratio * 100))

    if len(unique_names) >= 3:
        try:
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
            matrix = vectorizer.fit_transform(unique_names)
            similarity = cosine_similarity(matrix)
            for i in range(len(unique_names)):
                for j in range(i + 1, len(unique_names)):
                    score = float(similarity[i, j])
                    if score >= 0.82:
                        key = tuple(sorted((unique_names[i], unique_names[j])))
                        pairs[key] = max(pairs.get(key, 0), round(score * 100))
        except Exception:
            pass

    return [(left, right, score) for (left, right), score in pairs.items()]


def _duplicate_findings(df: pd.DataFrame, findings: list[dict]) -> None:
    row_count = len(df)
    exact_count = int(df.duplicated().sum())
    if exact_count:
        _finding(
            findings,
            category="Duplicates",
            severity="Medium",
            title="Exact duplicate records found",
            summary=f"{exact_count} exact duplicate record{'s' if exact_count != 1 else ''} detected.",
            evidence=[f"{exact_count} of {row_count} rows are repeated exactly."],
            recommendation="Remove exact duplicate records or confirm they represent legitimate repeated events.",
            affected_rows=exact_count,
            row_count=row_count,
        )

    name_col = _column_by_hints(df, ("full_name", "name"))
    if not name_col:
        return

    names = df[name_col].dropna().astype(str).str.strip()
    duplicate_names = names[names.duplicated(keep=False)]
    if not duplicate_names.empty:
        clusters = []
        for value, group in duplicate_names.groupby(duplicate_names):
            rows = ", ".join(_row_label(i) for i in group.index[:5])
            clusters.append(f"{value}: {len(group)} rows ({rows})")
        _finding(
            findings,
            category="Duplicates",
            severity="Medium",
            title="Duplicate customer identity values found",
            summary=f"{duplicate_names.nunique()} repeated {name_col} value{'s' if duplicate_names.nunique() != 1 else ''} found.",
            evidence=clusters[:5],
            recommendation="Review repeated identity values against stable customer identifiers before analysis.",
            affected_rows=int(len(duplicate_names)),
            row_count=row_count,
        )

    unique_names = names.drop_duplicates().tolist()
    fuzzy_pairs = _fuzzy_name_pairs(unique_names)
    if fuzzy_pairs:
        evidence = [f"{left} <> {right}: {score}% similarity" for left, right, score in fuzzy_pairs[:8]]
        _finding(
            findings,
            category="Duplicates",
            severity="Medium",
            title="Possible fuzzy duplicate customer names",
            summary=f"{len(fuzzy_pairs)} likely duplicate name pair{'s' if len(fuzzy_pairs) != 1 else ''} detected.",
            evidence=evidence,
            recommendation="Use fuzzy matching or human review to merge near-identical customer identities.",
            affected_rows=len(fuzzy_pairs),
            row_count=row_count,
        )


def _contact_findings(df: pd.DataFrame, findings: list[dict]) -> None:
    row_count = len(df)
    email_col = _column_by_hints(df, ("email", "e_mail"))
    if email_col:
        invalid = []
        for idx, value in df[email_col].dropna().items():
            text = str(value).strip()
            if text and not EMAIL_RE.match(text):
                invalid.append(f"{_row_label(idx)}: {text}")
        if invalid:
            _finding(
                findings,
                category="Validity",
                severity="High",
                title="Invalid email addresses detected",
                summary=f"{len(invalid)} invalid email address{'es' if len(invalid) != 1 else ''} detected.",
                evidence=invalid[:8],
                recommendation="Correct email formatting before using this data for outreach, matching, or account verification.",
                affected_rows=len(invalid),
                row_count=row_count,
                source_columns=[email_col],
                confidence=0.95,
            )

    phone_col = _column_by_hints(df, ("phone", "mobile", "contact_number"))
    if phone_col:
        invalid = []
        for idx, value in df[phone_col].dropna().items():
            digits = re.sub(r"\D", "", str(value))
            if digits and not 10 <= len(digits) <= 15:
                invalid.append(f"{_row_label(idx)}: {value}")
        if invalid:
            _finding(
                findings,
                category="Validity",
                severity="High",
                title="Invalid phone numbers detected",
                summary=f"{len(invalid)} invalid phone number{'s' if len(invalid) != 1 else ''} detected.",
                evidence=invalid[:8],
                recommendation="Normalize phone numbers to a consistent international or local format.",
                affected_rows=len(invalid),
                row_count=row_count,
                source_columns=[phone_col],
                confidence=0.92,
            )

    url_col = _column_by_hints(df, ("url", "website", "homepage", "link"))
    if url_col:
        invalid = []
        for idx, value in df[url_col].dropna().items():
            text = str(value).strip()
            if text and not URL_RE.match(text):
                invalid.append(f"{_row_label(idx)}: {text}")
        if invalid:
            _finding(
                findings,
                category="Validity",
                severity="Medium",
                title="Invalid URLs detected",
                summary=f"{len(invalid)} invalid URL value{'s' if len(invalid) != 1 else ''} detected.",
                evidence=invalid[:8],
                recommendation="Correct URL formatting to include a valid http or https scheme.",
                affected_rows=len(invalid),
                row_count=row_count,
                source_columns=[url_col],
                confidence=0.9,
            )


def _numeric_findings(df: pd.DataFrame, findings: list[dict]) -> None:
    row_count = len(df)
    age_col = _column_by_hints(df, ("age",))
    if age_col:
        ages = pd.to_numeric(df[age_col], errors="coerce")
        invalid = []
        for idx, value in ages.dropna().items():
            if value < 0 or value > 120:
                invalid.append(f"{_row_label(idx)}: {df.at[idx, age_col]}")
        if invalid:
            _finding(
                findings,
                category="Validity",
                severity="High",
                title="Impossible age values detected",
                summary=f"{len(invalid)} impossible age value{'s' if len(invalid) != 1 else ''} found.",
                evidence=invalid[:8],
                recommendation="Replace impossible ages with corrected values or mark them as missing for review.",
                affected_rows=len(invalid),
                row_count=row_count,
                source_columns=[age_col],
                confidence=0.96,
            )

    salary_col = _column_by_hints(df, ("salary", "income", "compensation", "pay"))
    if salary_col:
        values = pd.to_numeric(df[salary_col], errors="coerce").dropna()
        if len(values) >= 4:
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr if iqr else values.quantile(0.95)
            outliers = values[values > upper]
            if not outliers.empty:
                typical_max = values[values <= upper].max()
                if pd.isna(typical_max):
                    typical_max = values.quantile(0.95)
                evidence = [
                    f"{_row_label(idx)}: {salary_col} = {_safe_value(df.at[idx, salary_col])}"
                    for idx in outliers.index[:8]
                ]
                summary = (
                    f"{len(outliers)} extreme {salary_col} outlier{'s' if len(outliers) != 1 else ''} detected; "
                    f"typical values are at or below {float(typical_max):,.0f}."
                )
                _finding(
                    findings,
                    category="Outliers",
                    severity="Critical",
                    title="Critical salary outlier detected",
                    summary=summary,
                    evidence=evidence,
                    recommendation="Verify the salary source value before reporting averages, training models, or making compensation decisions.",
                    affected_rows=len(outliers),
                    row_count=row_count,
                    source_columns=[salary_col],
                    confidence=0.93,
                )
        negative = values[values < 0]
        if not negative.empty:
            _finding(
                findings,
                category="Business Logic",
                severity="High",
                title="Negative compensation values detected",
                summary=f"{len(negative)} negative {salary_col} value{'s' if len(negative) != 1 else ''} found.",
                evidence=[f"{_row_label(idx)}: {salary_col} = {_safe_value(df.at[idx, salary_col])}" for idx in negative.index[:8]],
                recommendation="Salary and compensation fields should not contain negative amounts unless explicitly documented.",
                affected_rows=len(negative),
                row_count=row_count,
                source_columns=[salary_col],
                confidence=0.97,
            )


def _date_findings(df: pd.DataFrame, findings: list[dict]) -> None:
    row_count = len(df)
    date_columns = [
        str(column)
        for column in df.columns
        if any(hint in str(column).lower() for hint in ("date", "dob", "birth"))
    ]
    parsed_by_column: dict[str, pd.Series] = {}
    for column in date_columns:
        raw = df[column].dropna()
        parsed = pd.to_datetime(raw, errors="coerce")
        parsed_by_column[column] = parsed
        invalid = [
            f"{_row_label(idx)}: {value}"
            for idx, value in raw.items()
            if str(value).strip() and pd.isna(parsed.loc[idx])
        ]
        if invalid:
            _finding(
                findings,
                category="Validity",
                severity="High",
                title=f"Invalid dates detected in {column}",
                summary=f"{len(invalid)} impossible or unparseable date value{'s' if len(invalid) != 1 else ''} found.",
                evidence=invalid[:8],
                recommendation="Correct invalid dates before age calculations, segmentation, or time-based reporting.",
                affected_rows=len(invalid),
                row_count=row_count,
            )

    age_col = _column_by_hints(df, ("age",))
    dob_col = _column_by_hints(df, ("date_of_birth", "dob", "birth"))
    if not age_col or not dob_col:
        return

    dob = pd.to_datetime(df[dob_col], errors="coerce")
    ages = pd.to_numeric(df[age_col], errors="coerce")
    today = datetime.now(timezone.utc).date()
    conflicts = []
    for idx in df.index:
        if pd.isna(dob.loc[idx]) or pd.isna(ages.loc[idx]):
            continue
        birth = dob.loc[idx].date()
        derived_age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        stated_age = float(ages.loc[idx])
        if stated_age < 0 or stated_age > 120 or abs(stated_age - derived_age) > 1:
            conflicts.append(
                f"{_row_label(idx)}: {age_col}={_safe_value(df.at[idx, age_col])}, "
                f"{dob_col}={_safe_value(df.at[idx, dob_col])}, expected age about {derived_age}"
            )
    if conflicts:
        _finding(
            findings,
            category="Business Logic",
            severity="High",
            title="Age does not match date of birth",
            summary=f"{len(conflicts)} age/date-of-birth conflict{'s' if len(conflicts) != 1 else ''} detected.",
            evidence=conflicts[:8],
            recommendation="Recalculate age from date of birth or remove one redundant field after confirming the source of truth.",
            affected_rows=len(conflicts),
            row_count=row_count,
        )


def _country_currency_findings(df: pd.DataFrame, findings: list[dict]) -> None:
    row_count = len(df)
    country_col = _column_by_hints(df, ("country", "nation"))
    if not country_col:
        return

    raw_countries = df[country_col].dropna().astype(str).str.strip()
    canonical_to_raw: dict[str, set[str]] = {}
    for value in raw_countries:
        canonical = COUNTRY_ALIASES.get(_norm(value), value.strip().title())
        canonical_to_raw.setdefault(canonical, set()).add(value)

    standardization = {
        canonical: sorted(values)
        for canonical, values in canonical_to_raw.items()
        if len(values) > 1
    }
    if standardization:
        evidence = [
            f"{canonical}: {', '.join(values)}"
            for canonical, values in standardization.items()
        ]
        _finding(
            findings,
            category="Standardization",
            severity="Medium",
            title="Country naming inconsistencies found",
            summary=f"{len(standardization)} country standardization issue{'s' if len(standardization) != 1 else ''} found.",
            evidence=evidence[:8],
            recommendation="Normalize country values to one canonical representation before grouping or reporting.",
            affected_rows=sum(len(values) for values in standardization.values()),
            row_count=row_count,
        )

    currency_col = _column_by_hints(df, ("currency",))
    if not currency_col:
        return

    mismatches = []
    for idx, row in df[[country_col, currency_col]].dropna().iterrows():
        country = COUNTRY_ALIASES.get(_norm(row[country_col]), str(row[country_col]).strip().title())
        currency = _norm(row[currency_col])
        expected = EXPECTED_CURRENCIES.get(country)
        if expected and currency not in expected:
            mismatches.append(
                f"{_row_label(idx)}: Country={row[country_col]}, Currency={row[currency_col]}"
            )
    if mismatches:
        _finding(
            findings,
            category="Semantic Consistency",
            severity="Medium",
            title="Country and currency mismatch found",
            summary=f"{len(mismatches)} country/currency mismatch{'es' if len(mismatches) != 1 else ''} detected.",
            evidence=mismatches[:8],
            recommendation="Validate currency from country rules or add a documented exception for multi-currency customers.",
            affected_rows=len(mismatches),
            row_count=row_count,
        )


def _missing_data_finding(profiles: list[dict], findings: list[dict], row_count: int) -> None:
    missing_columns = [
        profile for profile in profiles
        if float(profile.get("null_pct") or 0) > 0
    ]
    if not missing_columns:
        return

    max_null = max(float(profile.get("null_pct") or 0) for profile in missing_columns)
    if max_null <= 5:
        evidence = [
            f"{profile.get('column_name')}: {profile.get('null_pct')}% null"
            for profile in missing_columns[:8]
        ]
        _finding(
            findings,
            category="Completeness",
            severity="Low",
            title="Missing data level acceptable",
            summary=f"{len(missing_columns)} column{'s' if len(missing_columns) != 1 else ''} contain low missingness.",
            evidence=evidence,
            recommendation="Track these missing values, but they do not look severe at the current rate.",
            affected_rows=0,
            row_count=row_count,
        )


def _severity_rank(finding: dict) -> tuple[int, int]:
    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    return (order.get(finding.get("severity"), 9), -int(finding.get("affected_rows") or 0))


def build_quality_audit(df: pd.DataFrame, profiles: list[dict]) -> dict:
    findings: list[dict] = []
    row_count = len(df)

    _duplicate_findings(df, findings)
    _contact_findings(df, findings)
    _numeric_findings(df, findings)
    _date_findings(df, findings)
    _country_currency_findings(df, findings)
    _missing_data_finding(profiles, findings, row_count)

    for finding in findings:
        if not finding.get("confidence"):
            finding["confidence"] = 0.85

    findings.sort(key=_severity_rank)
    deductions = sum(SEVERITY_POINTS.get(finding["severity"], 0) for finding in findings)
    score = max(0, min(100, 100 - deductions))
    critical_count = sum(1 for item in findings if item["severity"] in {"Critical", "High"})
    warning_count = sum(1 for item in findings if item["severity"] == "Medium")
    suggestion_count = sum(1 for item in findings if item["severity"] == "Low")

    if score >= 85:
        summary = "Dataset is in strong shape, with only minor checks before use."
    elif score >= 70:
        summary = "Dataset is usable, but several quality issues should be cleaned before decisions or modeling."
    elif score >= 50:
        summary = "Dataset needs targeted cleaning before it can support reliable audit or ML work."
    else:
        summary = "Dataset has major quality risks and should not be used for decisions until the top issues are fixed."

    return {
        "score": int(score),
        "summary": summary,
        "critical_count": int(critical_count),
        "warning_count": int(warning_count),
        "suggestion_count": int(suggestion_count),
        "top_findings": findings[:6],
        "findings": findings,
    }
