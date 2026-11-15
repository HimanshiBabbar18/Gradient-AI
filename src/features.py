"""Feature selection on RAW columns (so the web app only asks for what matters).

Two complementary scorers, combined by average rank:
  * Mutual information        (filter, model-free, captures non-linear dependence)
  * Random-forest importance  (embedded, captures interactions)
K (number of kept features) is chosen with a cross-validated sweep on the TRAIN set only.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from . import config as C
from .data import build_preprocessor


def _ordinal_matrix(X: pd.DataFrame) -> pd.DataFrame:
    Xo = X.copy()
    cat = [c for c in Xo.columns if c in C.BINARY + C.NOMINAL]
    if cat:
        Xo[cat] = OrdinalEncoder().fit_transform(Xo[cat])
    return Xo.astype(float)


def rank_features(X_train: pd.DataFrame, y_train) -> pd.DataFrame:
    Xo = _ordinal_matrix(X_train)
    # every column here is discrete (small integer codes)
    mi = mutual_info_classif(Xo, y_train, discrete_features=True, random_state=C.SEED)
    rf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=C.SEED, n_jobs=-1
    ).fit(Xo, y_train)
    out = pd.DataFrame({"feature": Xo.columns, "mutual_info": mi, "rf_importance": rf.feature_importances_})
    out["mi_rank"] = out["mutual_info"].rank(ascending=False)
    out["rf_rank"] = out["rf_importance"].rank(ascending=False)
    out["avg_rank"] = (out["mi_rank"] + out["rf_rank"]) / 2
    return out.sort_values("avg_rank").reset_index(drop=True)


def choose_k(X_train: pd.DataFrame, y_train, ranking: pd.DataFrame,
             candidates=(5, 8, 10, 12, 15, 20), tol=0.01):
    """CV sweep over K; return the smallest K within `tol` of the best macro-F1."""
    n_all = X_train.shape[1]
    ks = sorted({k for k in candidates if k < n_all} | {n_all})
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    rows = []
    for k in ks:
        cols = ranking["feature"].head(k).tolist()
        pipe = Pipeline([
            ("prep", build_preprocessor(cols)),
            ("rf", RandomForestClassifier(200, class_weight="balanced", random_state=C.SEED, n_jobs=-1)),
        ])
        s = cross_val_score(pipe, X_train[cols], y_train, cv=cv, scoring="f1_macro")
        rows.append({"k": k, "cv_f1_macro": s.mean(), "cv_std": s.std()})
    sweep = pd.DataFrame(rows)
    best = sweep["cv_f1_macro"].max()
    k_star = int(sweep[sweep["cv_f1_macro"] >= best - tol]["k"].min())
    return k_star, sweep


def select_features(X_train: pd.DataFrame, y_train):
    ranking = rank_features(X_train, y_train)
    k, sweep = choose_k(X_train, y_train, ranking)
    selected = ranking["feature"].head(k).tolist()
    return selected, ranking, sweep
