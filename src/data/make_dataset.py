"""Build processed dataset from raw Telco churn CSV."""
from __future__ import annotations

import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW


def preprocess_data(
    input_path=None,
    output_path=None,
) -> pd.DataFrame:
    input_path = input_path or DATA_RAW
    output_path = output_path or DATA_PROCESSED

    df = pd.read_csv(input_path)
    print("Original data shape:", df.shape)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype(str).str.strip(), errors="coerce"
        )

    for col in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute TotalCharges for new customers (tenure=0) with MonthlyCharges
    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
        mask = df["TotalCharges"].isna()
        df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"]

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    y = df["Churn"] if "Churn" in df.columns else None
    X = df.drop(columns=["Churn"]) if y is not None else df
    X = pd.get_dummies(X, drop_first=False)

    out_df = X.copy()
    if y is not None:
        out_df["Churn"] = y

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")
    print(f"Processed data shape: {out_df.shape}")
    return out_df


if __name__ == "__main__":
    preprocess_data()