"""Traditional ML baselines (compared against the DNN). Each is a full
Pipeline (preprocessing + estimator) tuned with cross-validation on TRAIN only."""
import numpy as np
from scipy.stats import loguniform, randint
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from . import config as C
from .data import build_preprocessor


def train_baselines(X_train, y_train, columns, n_iter=15, cv_folds=3):
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=C.SEED)
    specs = {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            {"clf__C": loguniform(1e-3, 1e2)},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=C.SEED, n_jobs=-1),
            {"clf__n_estimators": randint(100, 500), "clf__max_depth": [None, 4, 6, 8, 12],
             "clf__min_samples_leaf": randint(1, 8), "clf__max_features": ["sqrt", "log2", 0.5]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=C.SEED),
            {"clf__n_estimators": randint(50, 300), "clf__learning_rate": loguniform(0.01, 0.3),
             "clf__max_depth": randint(2, 5), "clf__subsample": [0.7, 0.85, 1.0]},
        ),
    }
    fitted = {}
    for name, (est, space) in specs.items():
        pipe = Pipeline([("prep", build_preprocessor(columns)), ("clf", est)])
        search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=cv, scoring="f1_macro",
                                    random_state=C.SEED, n_jobs=1, refit=True)
        search.fit(X_train[columns], y_train)
        print(f"[baseline] {name:20s} cv macro-F1 = {search.best_score_:.4f}")
        fitted[name] = search.best_estimator_
    return fitted
