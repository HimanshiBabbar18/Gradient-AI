"""Metrics, plots and model-agnostic explainability."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_curve)
from sklearn.preprocessing import label_binarize

from . import config as C


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_proba is not None:
        from sklearn.metrics import roc_auc_score
        try:
            m["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            m["roc_auc_ovr"] = float("nan")
    return {k: float(v) for k, v in m.items()}


def print_report(name, y_true, y_pred):
    print(f"\n--- {name} : classification report (test set) ---")
    print(classification_report(y_true, y_pred, target_names=C.CLASS_NAMES, zero_division=0))


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(C.CLASS_NAMES, rotation=20); ax.set_yticklabels(C.CLASS_NAMES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_history(history: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["loss"], label="train"); axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["accuracy"], label="train"); axes[1].plot(history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_roc(y_true, y_proba, path):
    yb = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, name in enumerate(C.CLASS_NAMES):
        if yb[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yb[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("DNN ROC curves (one-vs-rest)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_comparison(df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df[["accuracy", "f1_macro"]].plot(kind="bar", ax=ax, rot=20)
    ax.set_ylim(0, 1); ax.set_title("Model comparison on held-out test set")
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def permutation_importance(predict_fn, X_test: pd.DataFrame, y_test, n_repeats=10, seed=C.SEED):
    """Model-agnostic importance on RAW columns: drop in macro-F1 when a column is shuffled."""
    rng = np.random.default_rng(seed)
    base = f1_score(y_test, predict_fn(X_test), average="macro")
    rows = []
    for col in X_test.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            drops.append(base - f1_score(y_test, predict_fn(Xp), average="macro"))
        rows.append({"feature": col, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def plot_importance(imp: pd.DataFrame, path, title="DNN permutation importance"):
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#4C72B0")
    ax.set_xlabel("Drop in macro-F1 when shuffled"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
