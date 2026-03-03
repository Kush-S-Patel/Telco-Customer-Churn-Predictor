import os
import pandas as pd

def preprocess_data():
    base_dir = os.path.dirname(__file__)
    input_path = os.path.join(base_dir, "../../data/raw/data.csv")
    output_path = os.path.join(base_dir, "../../data/processed/cleaned.csv")

    df = pd.read_csv(input_path)

    print("Original Data Shape:", df.shape)
    # Drop ID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Clean numeric columns
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")

    for col in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Target: Churn -> 0/1
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # One-hot encode categoricals (keep numeric as-is)
    X = df.drop(columns=["Churn"]) if "Churn" in df.columns else df
    y = df["Churn"] if "Churn" in df.columns else None

    X = pd.get_dummies(X, drop_first=False)

    out_df = X.copy()
    if y is not None:
        out_df["Churn"] = y

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Saved processed data to: {output_path}")
    print(f"Processed Data Shape: {out_df.shape}")
    return out_df

if __name__ == "__main__":
    preprocess_data()