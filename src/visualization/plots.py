"""Reusable plotting utilities for EDA and model evaluation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def setup_style():
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["figure.figsize"] = (10, 6)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_churn_distribution(df: pd.DataFrame, target: str, path: Path):
    setup_style()
    counts = df[target].value_counts().sort_index()
    labels = ["No Churn", "Churn"] if target == "Churn" and counts.index.tolist() == [0, 1] else counts.index.astype(str)
    if target == "Churn" and set(counts.index) <= {0, 1}:
        labels = ["No Churn (0)", "Churn (1)"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"][: len(counts)])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}\n({100*val/len(df):.1f}%)",
                ha="center", va="bottom", fontsize=11)
    ax.set_title("Customer Churn Distribution")
    ax.set_ylabel("Count")
    _save(fig, path)


def plot_numerical_by_churn(df: pd.DataFrame, cols: list[str], target: str, path: Path):
    setup_style()
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        if col not in df.columns:
            continue
        sns.boxplot(data=df, x=target, y=col, ax=ax, palette=["#4C72B0", "#C44E52"])
        ax.set_xticklabels(["No Churn", "Churn"] if target == "Churn" else ax.get_xticklabels())
        ax.set_title(f"{col} by Churn")
    fig.suptitle("Numerical Features vs Churn", y=1.02)
    fig.tight_layout()
    _save(fig, path)


def plot_correlation_heatmap(df: pd.DataFrame, path: Path, top_n: int = 20):
    setup_style()
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] > top_n:
        if "Churn" in numeric.columns:
            corr_target = numeric.corr()["Churn"].abs().sort_values(ascending=False)
            cols = corr_target.head(top_n).index.tolist()
            numeric = numeric[cols]
        else:
            numeric = numeric.iloc[:, :top_n]
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, ax=ax, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap (Top Predictors)")
    _save(fig, path)


def plot_categorical_churn_rates(df: pd.DataFrame, cat_cols: list[str], target: str, path: Path):
    setup_style()
    n = min(len(cat_cols), 6)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols[:n]):
        if col not in df.columns:
            continue
        rates = df.groupby(col)[target].mean().sort_values(ascending=False)
        rates.plot(kind="barh", ax=axes[i], color="#C44E52")
        axes[i].set_title(f"Churn rate by {col}")
        axes[i].set_xlabel("Churn rate")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Churn Rate by Categorical Features", y=1.02)
    fig.tight_layout()
    _save(fig, path)


def plot_roc_curve(y_true, y_proba_dict: dict, path: Path):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in y_proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_pr_curve(y_true, y_proba_dict: dict, path: Path):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in y_proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ax.plot(recall, precision, lw=2, label=name)
    baseline = y_true.mean()
    ax.axhline(baseline, color="k", linestyle="--", label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves")
    ax.legend(loc="upper right")
    _save(fig, path)


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str = "Confusion Matrix"):
    setup_style()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred: Stay", "Pred: Churn"],
                yticklabels=["Actual: Stay", "Actual: Churn"])
    ax.set_title(title)
    _save(fig, path)


def plot_feature_importance(importances: pd.Series, path: Path, top_k: int = 15):
    setup_style()
    top = importances.sort_values(ascending=True).tail(top_k)
    fig, ax = plt.subplots(figsize=(10, 7))
    top.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(f"Top {top_k} Feature Importances")
    ax.set_xlabel("Importance")
    _save(fig, path)


def plot_model_comparison(metrics_df: pd.DataFrame, path: Path):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, title in zip(
        axes,
        ["roc_auc", "f1", "recall"],
        ["ROC-AUC", "F1 Score", "Recall (Churn class)"],
    ):
        if metric not in metrics_df.columns:
            continue
        metrics_df[metric].plot(kind="bar", ax=ax, color="#55A868", edgecolor="black")
        ax.set_title(title)
        ax.set_ylabel(metric.upper().replace("_", " "))
        ax.set_xticklabels(metrics_df.index, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)
    fig.suptitle("Test Set Model Comparison", y=1.02)
    fig.tight_layout()
    _save(fig, path)


def plot_learning_curve(train_sizes, train_scores, val_scores, path: Path):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    ax.plot(train_sizes, train_mean, "o-", color="#4C72B0", label="Train")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#4C72B0")
    ax.plot(train_sizes, val_mean, "o-", color="#C44E52", label="Validation")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#C44E52")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Learning Curve (5-fold CV)")
    ax.legend()
    _save(fig, path)


def plot_calibration(y_true, y_proba, path: Path, n_bins: int = 10):
    from sklearn.calibration import calibration_curve

    setup_style()
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Plot")
    ax.legend()
    _save(fig, path)
