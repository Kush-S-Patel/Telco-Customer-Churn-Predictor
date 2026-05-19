"""Load and prepare features for modeling."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PROCESSED, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def load_processed_data(path=None) -> pd.DataFrame:
    path = path or DATA_PROCESSED
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)
    return df


def prepare_xy(df: pd.DataFrame, target: str = TARGET_COLUMN):
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y


def get_feature_names(df: pd.DataFrame, target: str = TARGET_COLUMN) -> list[str]:
    return [c for c in df.columns if c != target]


def train_test_split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    X, y = prepare_xy(df)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
