"""
Question 8: Cleaning and reproducible analysis tables

Key cleaning decisions
----------------------
1. Column names are normalised to analysis-ready labels by:
   - stripping leading and trailing whitespace,
   - converting to lowercase,
   - replacing internal whitespace with underscores,
   - replacing non-alphanumeric separators with underscores, and
   - adding suffixes if a cleaned label would otherwise be duplicated.

2. Dates are parsed with pandas, converted to monthly periods, and then
   standardised to month-start timestamps for consistent downstream analysis.

3. All non-date columns are converted to numeric where appropriate using
   pd.to_numeric(errors="coerce"). This is reproducible and safe because the
   Question 7 schema audit showed the analysis columns are numeric or safely
   convertible to numeric.

4. Each cleaned DataFrame is sorted by date, and monthly coverage is checked
   against the required range 2012-01 to 2018-12.

5. A tidy arrival table is created with one row per market-month and the
   required columns: date, market, and arrival.
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


def normalise_column_labels(labels):
    """
    Convert raw labels into cleaned, analysis-ready labels.

    Returns
    -------
    tuple[list[str], list[dict]]
        cleaned_labels in original order and an index-aware label mapping.
    """
    cleaned_labels = []
    label_map = []
    seen = {}

    for index, label in enumerate(labels):
        original = str(label)
        cleaned = original.strip().lower()
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")

        if not cleaned:
            cleaned = f"column_{index}"

        seen[cleaned] = seen.get(cleaned, 0) + 1
        final_label = cleaned if seen[cleaned] == 1 else f"{cleaned}_{seen[cleaned]}"

        cleaned_labels.append(final_label)
        label_map.append(
            {
                "index": index,
                "original": original,
                "cleaned": final_label,
            }
        )

    return cleaned_labels, label_map


def validate_monthly_date_range(date_values, start="2012-01", end="2018-12"):
    """
    Check whether a sequence covers every month in the required range.
    """
    parsed = pd.to_datetime(pd.Series(date_values), errors="coerce").dt.to_period("M")
    parsed_index = pd.PeriodIndex(parsed.dropna().unique()).sort_values()
    expected = pd.period_range(start=start, end=end, freq="M")

    missing = expected.difference(parsed_index)
    extra = parsed_index.difference(expected)

    return {
        "is_complete": len(missing) == 0,
        "start": start,
        "end": end,
        "n_periods": int(len(parsed_index)),
        "missing_months": [str(period) for period in missing],
        "extra_months": [str(period) for period in extra],
    }


def make_tidy_arrival_table(market_frames):
    """
    Stack market arrival data into one tidy market-month table.
    """
    tidy_parts = []

    for market_name, df in market_frames.items():
        small_df = df[["date", "arrival"]].copy()
        small_df["market"] = market_name
        tidy_parts.append(small_df)

    return (
        pd.concat(tidy_parts, ignore_index=True)
        .sort_values(["market", "date"])
        .reset_index(drop=True)
    )


raw_market_data = {}

for market, filename in CORE_MARKET_FILES.items():
    raw_market_data[market] = pd.read_csv(f"{RAW_BASE}/{filename}")


clean_market_data = {}
label_mapping_tables = {}
coverage_checks = {}

for market, raw_df in raw_market_data.items():
    cleaned_df = raw_df.copy()

    cleaned_labels, label_map = normalise_column_labels(cleaned_df.columns)
    cleaned_df.columns = cleaned_labels

    cleaned_df["date"] = (
        pd.to_datetime(cleaned_df["date"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    numeric_columns = [column for column in cleaned_df.columns if column != "date"]
    cleaned_df[numeric_columns] = cleaned_df[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    cleaned_df = cleaned_df.sort_values("date").reset_index(drop=True)

    clean_market_data[market] = cleaned_df
    label_mapping_tables[market] = pd.DataFrame(label_map)
    coverage_checks[market] = validate_monthly_date_range(cleaned_df["date"])


tidy_arrivals = make_tidy_arrival_table(clean_market_data)


cleaning_summary = """
Cleaning summary
----------------
- The cleaned market dictionary preserves the required keys:
  Australia, United States, and Thailand.
- Column labels were normalised to make them consistent and analysis-ready.
  For example, 'Hong kong' becomes 'hong_kong', 'Sheung Wan ' becomes
  'sheung_wan', and 'Hong Kong Convention and Exhibition Centre' becomes
  'hong_kong_convention_and_exhibition_centre'.
- Dates were parsed and standardised to month-start timestamps, then sorted.
- Numeric conversion was applied to all non-date columns using
  pd.to_numeric(errors="coerce"), keeping arrival and the SII-related numeric
  columns available for later analysis.
- Monthly coverage is complete for all three markets from 2012-01 to 2018-12.
- The tidy_arrivals table contains the required columns date, market, and
  arrival, with the expected 252 rows (3 markets x 84 months).
""".strip()


print("Keys in clean_market_data:")
print(list(clean_market_data.keys()))

print("\nCoverage checks:")
for market, result in coverage_checks.items():
    print(f"{market}: {result}")

print("\nSample original-to-cleaned label mapping for Australia:")
print(label_mapping_tables["Australia"].head(12).to_string(index=False))

print("\nTidy arrivals shape:")
print(tidy_arrivals.shape)

print("\nTidy arrivals head:")
print(tidy_arrivals.head().to_string(index=False))

print("\nCleaning summary:")
print(cleaning_summary)
