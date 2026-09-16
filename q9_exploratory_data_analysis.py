"""
Question 9: Exploratory data analysis and cross-market comparison

This script:
- loads and cleans the three required market files,
- creates the required annual_summary table,
- computes first-to-last change, volatility, seasonality, and a simple
  search-intensity comparison,
- saves clearly labelled comparison figures, and
- provides a short evidence-based written comparison.
"""

from __future__ import annotations

import re

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.use("Agg")


RAW_BASE = "https://raw.githubusercontent.com/tulip-lab/open-data/main/HK2012-2018"

CORE_MARKET_FILES = {
    "Australia": "Australia.csv",
    "United States": "United_States.csv",
    "Thailand": "Thailand.csv",
}


def normalise_column_labels(labels):
    """
    Convert raw labels into cleaned, analysis-ready labels.
    """
    cleaned_labels = []
    seen = {}

    for index, label in enumerate(labels):
        cleaned = str(label).strip().lower()
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")

        if not cleaned:
            cleaned = f"column_{index}"

        seen[cleaned] = seen.get(cleaned, 0) + 1
        final_label = cleaned if seen[cleaned] == 1 else f"{cleaned}_{seen[cleaned]}"
        cleaned_labels.append(final_label)

    return cleaned_labels


def load_clean_market_data():
    """
    Load the required market files and apply the Question 8 cleaning logic.
    """
    clean_market_data = {}

    for market, filename in CORE_MARKET_FILES.items():
        df = pd.read_csv(f"{RAW_BASE}/{filename}")
        df.columns = normalise_column_labels(df.columns)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

        numeric_columns = [column for column in df.columns if column != "date"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

        df = df.sort_values("date").reset_index(drop=True)
        clean_market_data[market] = df

    return clean_market_data


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


clean_market_data = load_clean_market_data()
tidy_arrivals = make_tidy_arrival_table(clean_market_data)

tidy_arrivals["year"] = tidy_arrivals["date"].dt.year
tidy_arrivals["month"] = tidy_arrivals["date"].dt.month


# Required object: annual_summary
annual_summary = (
    tidy_arrivals.groupby(["market", "year"], as_index=False)
    .agg(
        annual_total_arrival=("arrival", "sum"),
        annual_mean_arrival=("arrival", "mean"),
        annual_sd_arrival=("arrival", "std"),
        first_month_arrival=("arrival", "first"),
        last_month_arrival=("arrival", "last"),
    )
)


first_last_change = (
    tidy_arrivals.sort_values("date")
    .groupby("market", as_index=False)
    .agg(
        first_arrival=("arrival", "first"),
        last_arrival=("arrival", "last"),
    )
)
first_last_change["absolute_change"] = (
    first_last_change["last_arrival"] - first_last_change["first_arrival"]
)
first_last_change["pct_change"] = (
    first_last_change["absolute_change"] / first_last_change["first_arrival"] * 100
)


volatility_summary = (
    tidy_arrivals.groupby("market", as_index=False)
    .agg(
        mean_arrival=("arrival", "mean"),
        sd_arrival=("arrival", "std"),
        min_arrival=("arrival", "min"),
        max_arrival=("arrival", "max"),
    )
)
volatility_summary["cv"] = (
    volatility_summary["sd_arrival"] / volatility_summary["mean_arrival"]
)
volatility_summary["range"] = (
    volatility_summary["max_arrival"] - volatility_summary["min_arrival"]
)


seasonal_summary = (
    tidy_arrivals.groupby(["market", "month"], as_index=False)
    .agg(
        mean_monthly_arrival=("arrival", "mean"),
        sd_monthly_arrival=("arrival", "std"),
    )
)


# Simple search-intensity pattern summary.
# This uses the monthly mean across all cleaned SII columns as a compact
# descriptive measure for comparison. It is useful for EDA, but it should be
# interpreted cautiously because it aggregates many different search topics.
search_pattern_rows = []

for market, df in clean_market_data.items():
    sii_columns = [column for column in df.columns if column not in ("date", "arrival")]
    temp = df[["date", "arrival"] + sii_columns].copy()
    temp["mean_search_intensity"] = temp[sii_columns].mean(axis=1)

    search_pattern_rows.append(
        {
            "market": market,
            "mean_search_intensity": temp["mean_search_intensity"].mean(),
            "sd_search_intensity": temp["mean_search_intensity"].std(),
            "arrival_search_corr": temp["arrival"].corr(temp["mean_search_intensity"]),
        }
    )

search_pattern_summary = pd.DataFrame(search_pattern_rows)


# Figure 1: monthly arrivals time-series comparison
plt.figure(figsize=(12, 6))
for market, market_df in tidy_arrivals.groupby("market"):
    plt.plot(market_df["date"], market_df["arrival"], marker="o", linewidth=1.8, label=market)

plt.title("Monthly arrivals comparison across Australia, United States, and Thailand")
plt.xlabel("Month")
plt.ylabel("Arrival")
plt.legend(title="Market")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("q9_monthly_arrivals_comparison.png", dpi=200, bbox_inches="tight")
plt.close()


# Figure 2: seasonal comparison by calendar month
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.figure(figsize=(12, 6))
for market, market_df in seasonal_summary.groupby("market"):
    plt.plot(
        market_df["month"],
        market_df["mean_monthly_arrival"],
        marker="o",
        linewidth=2,
        label=market,
    )

plt.title("Seasonal profile: average arrival by calendar month")
plt.xlabel("Calendar month")
plt.ylabel("Average monthly arrival")
plt.xticks(range(1, 13), month_labels)
plt.legend(title="Market")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("q9_seasonal_arrivals_by_month.png", dpi=200, bbox_inches="tight")
plt.close()


q9_written_comparison = """
Across the three required markets, the United States has by far the highest
arrival volume, with a mean monthly arrival of about 99,257 compared with
49,335 for Australia and 44,969 for Thailand. This difference is visible in
 the monthly time-series figure and is also reflected in the annual_summary
table, where the United States remains the largest market throughout the
period. In terms of growth, Thailand shows the strongest improvement from the
first month to the last month, increasing from 26,986 to 68,486 arrivals
(+153.8%). The United States also grows meaningfully from 86,116 to 117,517
(+36.5%), while Australia is relatively flat overall, changing from 60,116 to
61,122 (+1.7%).

The markets also differ in volatility and seasonality. Thailand is the most
volatile market on a relative basis, with the highest standard deviation
(11,397) relative to its mean and the largest coefficient of variation
(0.253). The United States has the largest absolute range because its volume is
largest, but it is the least volatile relative to its scale (CV 0.143),
suggesting more stable demand. The seasonal summary shows that the United
States tends to peak around March-April and again in October-November, while
Thailand peaks strongly in March-April and especially December. Australia shows
smaller but clear seasonal highs around January, April, October, and December,
with weaker months such as February and August. Using a simple average across
search-intensity columns, Australia has the highest average search-intensity
level, but Thailand shows the strongest arrival-search correlation, suggesting
that search behaviour may move more closely with arrivals there than in the
other two markets. Because this search measure averages many topics, it should
be treated as descriptive rather than causal evidence.
""".strip()


print("Figure saved: q9_monthly_arrivals_comparison.png")
print("Figure saved: q9_seasonal_arrivals_by_month.png")

print("\nannual_summary:")
print(annual_summary.to_string(index=False))

print("\nfirst_last_change:")
print(first_last_change.round(2).to_string(index=False))

print("\nvolatility_summary:")
print(volatility_summary.round(4).to_string(index=False))

print("\nseasonal_summary:")
print(seasonal_summary.round(2).to_string(index=False))

print("\nsearch_pattern_summary:")
print(search_pattern_summary.round(4).to_string(index=False))

print("\nQ9 written comparison:")
print(q9_written_comparison)
