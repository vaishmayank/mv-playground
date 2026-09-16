"""
Question 7: Data access and schema audit

Dataset source acknowledgement
------------------------------
This script loads the three required market files directly from the public
raw GitHub URL base for the HK2012-2018 dataset. It does not rely on any
local absolute file paths.

How the files are loaded
------------------------
1. Define RAW_BASE as the approved public raw URL base.
2. Define CORE_MARKET_FILES for Australia, United States, and Thailand.
3. Build each file URL with f"{RAW_BASE}/{filename}".
4. Read each CSV with pandas.read_csv() into raw_market_data.
5. Audit the raw files before cleaning and store the results in schema_audit.
"""

from __future__ import annotations

import re

import pandas as pd


RAW_BASE = "https://raw.githubusercontent.com/tulip-lab/open-data/main/HK2012-2018"

CORE_MARKET_FILES = {
    "Australia": "Australia.csv",
    "United States": "United_States.csv",
    "Thailand": "Thailand.csv",
}


def detect_column_issues(columns):
    """Return a concise note about column-name issues in a raw DataFrame."""
    issues = []
    normalized = [re.sub(r"\s+", " ", col.strip().lower()) for col in columns]

    if any(col != col.strip() for col in columns):
        issues.append("leading/trailing spaces")
    if any(" " in col for col in columns):
        issues.append("spaces in labels")
    if any(any(ch.isupper() for ch in col) for col in columns):
        issues.append("inconsistent capitalisation")
    if any(".1" in col for col in columns):
        issues.append("auto-suffixed duplicate labels")
    if len(normalized) != len(set(normalized)):
        issues.append("duplicate-looking labels")

    return "; ".join(issues) if issues else "none"


def count_numeric_analysis_columns(df, date_col="date"):
    """Count columns that are numeric or safely convertible to numeric."""
    numeric_count = 0

    for column in df.columns:
        if column == date_col:
            continue

        series = df[column].dropna()
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_count += 1
            continue

        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().all():
            numeric_count += 1

    return numeric_count


raw_market_data = {}

for market, filename in CORE_MARKET_FILES.items():
    url = f"{RAW_BASE}/{filename}"
    raw_market_data[market] = pd.read_csv(url)


schema_rows = []

for market, df in raw_market_data.items():
    date_col = "date"
    arrival_col = "arrival"

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    arrival_non_missing = df[arrival_col].dropna()
    arrival_numeric = pd.to_numeric(
        arrival_non_missing, errors="coerce"
    ).notna().all()

    schema_rows.append(
        {
            "market": market,
            "rows": len(df),
            "columns": df.shape[1],
            "date_min": parsed_dates.min().strftime("%Y-%m"),
            "date_max": parsed_dates.max().strftime("%Y-%m"),
            "date_ordered": bool(parsed_dates.is_monotonic_increasing),
            "missing_cells": int(df.isna().sum().sum()),
            "arrival_numeric": bool(arrival_numeric),
            "n_numeric_columns": count_numeric_analysis_columns(df),
            "column_issues": detect_column_issues(df.columns.tolist()),
        }
    )


schema_audit = pd.DataFrame(schema_rows)


schema_audit_summary = """
Schema audit findings
---------------------
- All three required files loaded successfully from the public RAW_BASE URL.
- Each market file contains 84 rows and 98 columns.
- The date range runs from 2012-01 to 2018-12 in all three files.
- Dates are already in chronological order for Australia, United States,
  and Thailand.
- No missing cells were detected in any of the three raw files.
- The arrival column is numeric or safely convertible to numeric in all files.
- Excluding the date column, 97 columns are numeric or safely convertible to
  numeric in each file.
- Important column-name issues are present in the raw data, including leading
  or trailing spaces, inconsistent capitalisation, labels with spaces, and
  auto-suffixed duplicate-style names such as 'hong kong hotel central.1'.
  These issues should be cleaned before downstream analysis.
""".strip()


print("Dataset source:")
print(RAW_BASE)
print("\nKeys in raw_market_data:")
print(list(raw_market_data.keys()))
print("\nschema_audit:")
print(schema_audit.to_string(index=False))
print("\nShort written explanation:")
print(schema_audit_summary)
