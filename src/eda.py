"""Exploratory data analysis: prints key stats and saves figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as C

sns.set_theme(style="whitegrid")


def run_eda(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== EDA ===")
    print(f"shape: {df.shape}")
    print("class distribution (grade band):")
    print(df["grade_class"].map(dict(enumerate(C.CLASS_NAMES))).value_counts().to_string())
    print(df[["G1", "G2", "G3", "absences", "studytime", "failures"]].describe().round(2).to_string())

    # 1. G3 histogram with band boundaries
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[C.TARGET], bins=21, ax=ax, color="#4C72B0")
    for x in (9.5, 14.5):
        ax.axvline(x, color="red", ls="--")
    ax.set_title("Final grade (G3) distribution and class boundaries")
    fig.tight_layout(); fig.savefig(out_dir / "01_g3_distribution.png", dpi=130); plt.close(fig)

    # 2. class balance
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["grade_class"].value_counts().sort_index()
    ax.bar(C.CLASS_NAMES, counts.values, color=["#C44E52", "#4C72B0", "#55A868"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center")
    ax.set_title("Target classes (imbalanced -> class weights + macro-F1)")
    fig.tight_layout(); fig.savefig(out_dir / "02_class_balance.png", dpi=130); plt.close(fig)

    # 3. correlation heatmap (numeric features + G3)
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[C.NUMERIC + [C.TARGET]].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax)
    ax.set_title("Correlation matrix (numeric features vs G3)")
    fig.tight_layout(); fig.savefig(out_dir / "03_correlation.png", dpi=130); plt.close(fig)

    # 4. G3 vs key behavioural variables
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["failures", "studytime", "higher"]):
        sns.boxplot(data=df, x=col, y=C.TARGET, ax=ax, hue=col, legend=False, palette="Set2")
        ax.set_title(f"G3 by {col}")
    fig.tight_layout(); fig.savefig(out_dir / "04_g3_vs_key_features.png", dpi=130); plt.close(fig)

    # 5. absences vs G3 and G2 vs G3
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.scatterplot(data=df, x="absences", y=C.TARGET, hue="grade_class", palette="deep", ax=axes[0], legend=False)
    axes[0].set_title("Absences vs G3")
    sns.scatterplot(data=df, x="G2", y=C.TARGET, hue="grade_class", palette="deep", ax=axes[1], legend=False)
    axes[1].set_title("G2 vs G3 (strong leakage-like signal)")
    fig.tight_layout(); fig.savefig(out_dir / "05_absences_g2_vs_g3.png", dpi=130); plt.close(fig)

    # 6. class share by categorical features
    cats = ["sex", "address", "schoolsup", "internet", "romantic", "Mjob"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), cats):
        (df.groupby(col)["grade_class"].value_counts(normalize=True).unstack()
           .rename(columns=dict(enumerate(C.CLASS_NAMES))).plot(kind="bar", stacked=True, ax=ax, legend=False))
        ax.set_title(f"Class share by {col}"); ax.set_xlabel("")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "06_class_share_by_category.png", dpi=130); plt.close(fig)

    top = corr[C.TARGET].drop(C.TARGET).abs().sort_values(ascending=False).head(5)
    print("top |corr| with G3:\n" + top.round(3).to_string())
    print(f"[eda] figures saved to {out_dir}")
