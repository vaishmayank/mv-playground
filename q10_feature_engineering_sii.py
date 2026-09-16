"""
Question 10: Feature engineering and SII analysis

This script:
- loads and cleans the three required market files,
- builds interpretable arrival-derived features within each market,
- transparently selects at least three original SII features,
- summarises the selected SII features by market, and
- provides a short non-causal interpretation of the observed relationships.
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


def detect_local_extrema(values):
    """
    Return zero-based indices for strict local peaks and valleys.
    """
    peaks = []
    valleys = []

    for index in range(1, len(values) - 1):
        left = values[index - 1]
        current = values[index]
        right = values[index + 1]

        if left < current and current > right:
            peaks.append(index)
        elif left > current and current < right:
            valleys.append(index)

    return {"peaks": peaks, "valleys": valleys}


def make_feature_table(clean_market_data):
    """
    Create the required arrival-derived feature table within each market.
    """
    feature_parts = []

    for market, df in clean_market_data.items():
        temp = df[["date", "arrival"]].copy()
        temp["market"] = market
        temp["year"] = temp["date"].dt.year
        temp["month"] = temp["date"].dt.month
        temp["arrival_diff"] = temp["arrival"].diff()
        temp["arrival_pct_change"] = temp["arrival"].pct_change() * 100
        temp["arrival_roll3_mean"] = temp["arrival"].rolling(window=3, min_periods=3).mean()
        temp["arrival_roll12_mean"] = temp["arrival"].rolling(window=12, min_periods=12).mean()
        temp["arrival_volatility_3"] = temp["arrival"].rolling(window=3, min_periods=3).std()

        extrema = detect_local_extrema(temp["arrival"].tolist())
        peak_index_set = set(extrema["peaks"])
        valley_index_set = set(extrema["valleys"])

        temp["is_local_peak"] = [index in peak_index_set for index in range(len(temp))]
        temp["is_local_valley"] = [index in valley_index_set for index in range(len(temp))]

        feature_parts.append(temp)

    feature_table = pd.concat(feature_parts, ignore_index=True)
    return feature_table[
        [
            "date",
            "market",
            "arrival",
            "year",
            "month",
            "arrival_diff",
            "arrival_pct_change",
            "arrival_roll3_mean",
            "arrival_roll12_mean",
            "arrival_volatility_3",
            "is_local_peak",
            "is_local_valley",
        ]
    ]


clean_market_data = load_clean_market_data()


# Required object: feature_table
feature_table = make_feature_table(clean_market_data)


# Transparent SII feature-selection rule:
# 1. Start with original cleaned SII columns only (exclude date and arrival).
# 2. Restrict to a pre-specified candidate pool of interpretable travel-planning
#    or trip-context search terms.
# 3. Rank the candidates by average absolute Pearson correlation with monthly
#    arrivals across the three required markets.
# 4. Select the top three candidates.
candidate_sii_pool = [
    "skyscanner",
    "flights_to_hong_kong",
    "hong_kong_hotel",
    "hong_kong_accommodation",
    "cathay_pacific",
    "agoda_hong_kong",
    "hong_kong_airport",
    "hong_kong_map",
    "hong_kong_weather",
    "hong_kong_disneyland",
]


sii_selection_rows = []

for feature in candidate_sii_pool:
    market_correlations = []

    for market, df in clean_market_data.items():
        market_correlations.append(abs(df["arrival"].corr(df[feature])))

    sii_selection_rows.append(
        {
            "feature": feature,
            "avg_abs_corr": sum(market_correlations) / len(market_correlations),
        }
    )

sii_selection_table = pd.DataFrame(sii_selection_rows).sort_values(
    "avg_abs_corr", ascending=False
).reset_index(drop=True)


# Required object: selected_sii_features
selected_sii_features = sii_selection_table.head(3)["feature"].tolist()


# Required object: sii_summary
sii_summary_rows = []

for market, df in clean_market_data.items():
    for feature in selected_sii_features:
        relationship_value = df["arrival"].corr(df[feature])

        if abs(relationship_value) < 0.1:
            direction = "weak or no clear linear"
        elif relationship_value > 0:
            direction = "positive"
        elif relationship_value < 0:
            direction = "negative"
        else:
            direction = "no clear linear"

        interpretation_note = (
            f"{direction.capitalize()} relationship in this market; "
            "the measure suggests association, not causation, and may reflect "
            "shared trends, seasonality, or other confounding factors."
        )

        sii_summary_rows.append(
            {
                "market": market,
                "feature": feature,
                "feature_mean": df[feature].mean(),
                "feature_sd": df[feature].std(),
                "relationship_measure": "pearson_correlation",
                "relationship_value": relationship_value,
                "interpretation_note": interpretation_note,
            }
        )

sii_summary = pd.DataFrame(sii_summary_rows)


# Comparison figure: correlation values by market and selected SII feature
correlation_plot_data = sii_summary.pivot(
    index="feature", columns="market", values="relationship_value"
)

ax = correlation_plot_data.plot(
    kind="bar",
    figsize=(12, 6),
    rot=20,
)
ax.set_title("Selected SII feature correlations with arrivals across markets")
ax.set_xlabel("Selected SII feature")
ax.set_ylabel("Pearson correlation with arrival")
ax.legend(title="Market")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("q10_sii_arrival_relationships.png", dpi=200, bbox_inches="tight")
plt.close()


q10_sii_interpretation = """
The selected SII features were chosen using a transparent rule applied to the
original cleaned search columns before any arrival-derived features were added.
I first restricted attention to interpretable travel-planning or trip-context
search terms such as booking platforms, transport, attraction, map, and weather
queries. I then ranked these candidate features by their average absolute
Pearson correlation with monthly arrivals across Australia, the United States,
and Thailand. Using this rule, the top three features were
hong_kong_weather, skyscanner, and hong_kong_disneyland. These features are
traceable to the cleaned original CSV columns and were selected because they
showed relatively strong or consistent co-movement with arrivals while still
being easy to explain in a tourism context.

The observed relationships suggest association, not causation. For example,
hong_kong_weather is more strongly positively correlated with arrivals in
Thailand and Australia than in the United States, while skyscanner shows mixed
signs across markets, suggesting that search timing and travel behaviour may
differ by source market. Hong_kong_disneyland shows moderate positive
correlations across all three markets, making it a comparatively stable tourism
signal. However, none of these relationships proves that higher search
intensity causes higher arrivals. The correlations may also reflect common
seasonal patterns, long-run trends, promotions, holidays, macroeconomic
conditions, or other confounding influences.
""".strip()


print("Feature-selection rule: top 3 features by average absolute Pearson")
print("correlation within a pre-specified interpretable SII candidate pool.")

print("\nselected_sii_features:")
print(selected_sii_features)

print("\nfeature_table:")
print(feature_table.head(15).to_string(index=False))

print("\nsii_selection_table:")
print(sii_selection_table.round(4).to_string(index=False))

print("\nsii_summary:")
print(sii_summary.round(4).to_string(index=False))

print("\nFigure saved: q10_sii_arrival_relationships.png")

print("\nQ10 SII interpretation:")
print(q10_sii_interpretation)
