"""End-to-end pipeline:  data -> EDA -> split -> feature selection -> DNN tuning ->
training -> evaluation -> comparison with classical ML -> save artifacts for the app.

Usage:
    python train.py                    # Portuguese dataset, both scenarios
    python train.py --dataset mat      # Mathematics dataset
    python train.py --trials 25        # more tuning trials
    python train.py --scenarios with_grades
"""
import argparse
import time

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src import data as D
from src import dnn, evaluate as E
from src.baselines import train_baselines
from src.eda import run_eda
from src.selection import select_features


def feature_metadata(X: pd.DataFrame, selected):
    """Everything the Streamlit app needs to build its input form."""
    meta = []
    for c in selected:
        label, help_ = C.LABELS[c]
        item = {"name": c, "label": label, "help": help_}
        if c in C.NUMERIC:
            item.update(type="numeric", min=int(X[c].min()), max=int(X[c].max()),
                        default=int(round(X[c].median())))
        else:
            opts = sorted(X[c].unique().tolist())
            item.update(type="categorical", options=opts, default=X[c].mode()[0])
        meta.append(item)
    return meta


def run_scenario(name, df, args):
    print(f"\n{'=' * 70}\nSCENARIO: {name}\n{'=' * 70}")
    out = C.ARTIFACT_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = D.split(df, drop_cols=C.SCENARIOS[name])
    print(f"[split] train={len(X_train)} test={len(X_test)} features={X_train.shape[1]}")

    # ---- feature selection (train only) ----
    selected, ranking, sweep = select_features(X_train, y_train)
    ranking.to_csv(out / "feature_ranking.csv", index=False)
    sweep.to_csv(out / "k_sweep.csv", index=False)
    print(f"[select] kept {len(selected)}/{X_train.shape[1]} features: {selected}")
    Xtr, Xte = X_train[selected], X_test[selected]

    # ---- classical ML baselines ----
    baselines = train_baselines(X_train, y_train, selected, n_iter=args.ml_iter)

    # ---- DNN: tune -> final fit ----
    t0 = time.time()
    prep = D.build_preprocessor(selected)
    best_hp, tune_df = dnn.random_search(Xtr, y_train, prep, n_trials=args.trials, folds=args.folds)
    tune_df.to_csv(out / "dnn_tuning_results.csv", index=False)
    print(f"[tune] best hyper-parameters: {best_hp}  ({time.time() - t0:.0f}s)")

    prep = D.build_preprocessor(selected)
    model, history = dnn.fit_final(Xtr, y_train, prep, best_hp)
    Xte_p = prep.transform(Xte)
    dnn_proba = dnn.predict_proba(model, Xte_p)
    dnn_pred = dnn_proba.argmax(axis=1)
    model.summary()

    # ---- evaluation ----
    results = {"DNN (Keras)": E.compute_metrics(y_test, dnn_pred, dnn_proba)}
    E.print_report("DNN", y_test, dnn_pred)
    E.plot_confusion(y_test, dnn_pred, f"DNN confusion matrix ({name})", out / "confusion_dnn.png")
    E.plot_history(history, out / "training_curves.png")
    E.plot_roc(y_test, dnn_proba, out / "roc_dnn.png")

    best_ml_pred = None
    for mname, pipe in baselines.items():
        pred = pipe.predict(Xte)
        results[mname] = E.compute_metrics(y_test, pred, pipe.predict_proba(Xte))
        E.print_report(mname, y_test, pred)
        if mname == "Random Forest":
            E.plot_confusion(y_test, pred, f"Random Forest confusion matrix ({name})", out / "confusion_rf.png")

    comp = pd.DataFrame(results).T.sort_values("f1_macro", ascending=False)
    comp.to_csv(out / "comparison.csv")
    E.plot_comparison(comp, out / "comparison.png")
    print("\n=== MODEL COMPARISON (held-out test set) ===")
    print(comp.round(4).to_string())

    # ---- explainability ----
    predict_raw = lambda Xraw: dnn.predict_proba(model, prep.transform(Xraw)).argmax(axis=1)
    imp = E.permutation_importance(predict_raw, Xte, np.asarray(y_test))
    imp.to_csv(out / "permutation_importance.csv", index=False)
    E.plot_importance(imp, out / "permutation_importance.png", f"DNN permutation importance ({name})")
    print("[explain] top features:", imp["feature"].head(5).tolist())

    # ---- save artifacts for the web app ----
    model.save(str(out / "dnn_model.keras"))
    joblib.dump(prep, out / "preprocessor.joblib")
    E.save_json({
        "scenario": name, "dataset": args.dataset, "selected_features": selected,
        "features": feature_metadata(df, selected), "class_names": C.CLASS_NAMES,
        "best_hyperparameters": best_hp, "test_metrics": results,
        "n_train": len(X_train), "n_test": len(X_test),
    }, out / "meta.json")
    print(f"[save] artifacts written to {out}")
    return comp.assign(scenario=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="por", choices=["por", "mat"])
    ap.add_argument("--trials", type=int, default=15, help="DNN random-search trials")
    ap.add_argument("--folds", type=int, default=3, help="CV folds for DNN tuning")
    ap.add_argument("--ml-iter", type=int, default=15, help="random-search iterations for ML baselines")
    ap.add_argument("--quick", action="store_true", help="fast smoke test (2 trials, 2 folds, 3 ML iters)")
    ap.add_argument("--scenarios", nargs="+", default=list(C.SCENARIOS), choices=list(C.SCENARIOS))
    args = ap.parse_args()
    if args.quick:
        args.trials, args.folds, args.ml_iter = 2, 2, 3

    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = D.add_target(D.clean(D.load_raw(args.dataset)))
    run_eda(df, C.ARTIFACT_DIR / "eda")

    summary = pd.concat([run_scenario(s, df, args) for s in args.scenarios])
    summary.to_csv(C.ARTIFACT_DIR / "summary.csv")
    print("\n\n######## FINAL SUMMARY ########")
    print(summary.round(4).to_string())
    print("\nNext step:  streamlit run app.py")


if __name__ == "__main__":
    main()
