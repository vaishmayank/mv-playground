"""
Prepare the Obesity dataset for clustering.

Steps:
  1. Load the raw CSV.
  2. Remove the class label 'NObeyesdad' from the feature set.
  3. Encode all non-numeric variables (binary, ordinal, nominal).
  4. Scale numeric features so no single feature dominates distances.

Run:  python3 prepare_obesity_for_clustering.py
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = "ObesityDataSet_raw_and_data_sinthetic.csv"
OUT_PATH = "obesity_prepared_for_clustering.csv"


def main() -> None:
    # 1. Load
    df = pd.read_csv(RAW_PATH)
    print(f"Raw shape: {df.shape}")

    # 2. Remove the class label -> unsupervised clustering uses features only.
    #    Keep it aside so clusters can be validated against it later.
    y = df["NObeyesdad"].copy()
    X = df.drop(columns=["NObeyesdad"])

    # 3. Define variable groups
    binary_yes_no = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    ordinal_cols = ["CAEC", "CALC"]          # no < Sometimes < Frequently < Always
    nominal_cols = ["MTRANS"]                # no natural order -> one-hot
    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

    # 3a. Binary encoding (0/1)
    X["Gender"] = X["Gender"].map({"Female": 0, "Male": 1})
    for col in binary_yes_no:
        X[col] = X[col].map({"no": 0, "yes": 1})

    # 3b. Ordinal encoding (preserve order)
    order = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    for col in ordinal_cols:
        X[col] = X[col].map(order)

    # 3c. One-hot encoding for nominal variable
    X = pd.get_dummies(X, columns=nominal_cols, prefix="MTRANS")
    # Convert the new boolean one-hot columns to integers (0/1)
    ohe_cols = [c for c in X.columns if c.startswith("MTRANS_")]
    X[ohe_cols] = X[ohe_cols].astype(int)

    # 4. Scale numeric features (mean 0, std 1)
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Report
    print(f"Prepared shape: {X.shape}")
    print(f"\nColumns ({len(X.columns)}):\n{list(X.columns)}")
    print(f"\nAll dtypes numeric? {all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)}")
    print(f"Any missing values? {X.isnull().any().any()}")
    print("\nFirst 5 rows:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(X.head())
    print("\nNumeric feature summary after scaling (mean~0, std~1):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(X[numeric_cols].describe().loc[["mean", "std", "min", "max"]].round(3))

    # Save prepared features
    X.to_csv(OUT_PATH, index=False)
    print(f"\nSaved prepared features to: {OUT_PATH}")


if __name__ == "__main__":
    main()
