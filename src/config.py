"""Project paths and training configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "data.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Churn"

# Default champion model hyperparameters (tuned on validation set)
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "logloss",
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 14,
    "min_samples_leaf": 4,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
