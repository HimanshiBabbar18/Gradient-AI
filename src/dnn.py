"""Deep Neural Network (Keras): architecture, training and hyper-parameter search."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras

from . import config as C

N_CLASSES = len(C.CLASS_NAMES)

# Search space for random search
SPACE = {
    "n_layers": [1, 2, 3],
    "units": [16, 32, 64, 128],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "l2": [0.0, 1e-4, 1e-3, 1e-2],
    "lr": [3e-4, 1e-3, 3e-3, 1e-2],
    "batch_size": [16, 32, 64],
    "activation": ["relu", "elu"],
}


def build_dnn(input_dim: int, hp: dict) -> keras.Model:
    """Dense -> BatchNorm -> Activation -> Dropout blocks (width halves each layer),
    softmax output over the 3 grade classes."""
    reg = keras.regularizers.l2(hp["l2"]) if hp["l2"] > 0 else None
    model = keras.Sequential(name="student_dnn")
    model.add(keras.layers.Input(shape=(input_dim,)))
    for i in range(hp["n_layers"]):
        units = max(8, hp["units"] // (2 ** i))
        model.add(keras.layers.Dense(units, kernel_regularizer=reg, kernel_initializer="he_normal"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(hp["activation"]))
        model.add(keras.layers.Dropout(hp["dropout"]))
    model.add(keras.layers.Dense(N_CLASSES, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _class_weights(y):
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(x) for c, x in zip(classes, w)}


def train_dnn(X_tr, y_tr, X_val, y_val, hp, epochs=300, verbose=0):
    """Train with class weights, early stopping and LR scheduling."""
    keras.utils.set_random_seed(C.SEED)
    model = build_dnn(X_tr.shape[1], hp)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]
    hist = model.fit(
        X_tr, np.asarray(y_tr), validation_data=(X_val, np.asarray(y_val)),
        epochs=epochs, batch_size=hp["batch_size"], class_weight=_class_weights(y_tr),
        callbacks=callbacks, verbose=verbose,
    )
    return model, hist.history


def predict_proba(model, X) -> np.ndarray:
    return model.predict(X, batch_size=256, verbose=0)


def random_search(X_train: pd.DataFrame, y_train, preprocessor, n_trials=15, folds=3, seed=C.SEED):
    """Random search scored by stratified k-fold macro-F1 on the TRAIN set.
    The preprocessor is re-fitted inside every fold (no leakage)."""
    rng = np.random.default_rng(seed)
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    y_train = np.asarray(y_train)
    results = []
    for t in range(n_trials):
        hp = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        hp = {k: (float(v) if isinstance(v, (float, np.floating)) else
                  int(v) if isinstance(v, (int, np.integer)) else v) for k, v in hp.items()}
        scores = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            prep = clone(preprocessor)
            Xtr = prep.fit_transform(X_train.iloc[tr_idx])
            Xva = prep.transform(X_train.iloc[va_idx])
            model, _ = train_dnn(Xtr, y_train[tr_idx], Xva, y_train[va_idx], hp, epochs=150)
            pred = predict_proba(model, Xva).argmax(axis=1)
            scores.append(f1_score(y_train[va_idx], pred, average="macro"))
            keras.backend.clear_session()
        row = {**hp, "cv_f1_macro": float(np.mean(scores)), "cv_std": float(np.std(scores))}
        results.append(row)
        print(f"[tune] trial {t + 1:2d}/{n_trials}  cv macro-F1={row['cv_f1_macro']:.4f}  {hp}")
    res = pd.DataFrame(results).sort_values("cv_f1_macro", ascending=False).reset_index(drop=True)
    best = {k: res.loc[0, k] for k in SPACE}
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    return best, res


def fit_final(X_train: pd.DataFrame, y_train, preprocessor, hp):
    """Fit preprocessor on the full train set, hold out VAL_SIZE for early stopping."""
    Xp = preprocessor.fit_transform(X_train)
    y = np.asarray(y_train)
    Xtr, Xva, ytr, yva = train_test_split(Xp, y, test_size=C.VAL_SIZE, stratify=y, random_state=C.SEED)
    model, history = train_dnn(Xtr, ytr, Xva, yva, hp, epochs=400)
    return model, history
