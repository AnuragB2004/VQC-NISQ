"""Classical baselines.

mode 'matched' (primary, Table 6): the six model families on exactly the same 15
stratified splits and the same leakage-free PCA features (dimension 4 and 8) as the VQC.
mode 'nested' (secondary, Table 5): 5 repetitions x 5-fold outer CV on the full
min-max-scaled feature set, with hyperparameters chosen by inner 5-fold CV.

Logistic regression (l2, C=1) and linear SVM (C=1) are fixed; RBF SVM, random forest, MLP
and XGBoost are grid-searched (inner stratified 5-fold CV on the training partition only)
over the grids stated in the manuscript. scikit-learn's MLPClassifier has no dropout, so
the MLP grid is capacity x learning rate only.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import common as C

warnings.filterwarnings("ignore", category=ConvergenceWarning)
MODE = sys.argv[1]
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results"); OUT.mkdir(parents=True, exist_ok=True)
N_JOBS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

GRIDS = {
    "RBF SVM": (SVC(kernel="rbf"), {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 1e-3, 1e-2]}),
    "Random Forest": (RandomForestClassifier(n_jobs=1, random_state=0),
                      {"n_estimators": [100, 200, 500], "max_depth": [5, 10, 20, None], "min_samples_leaf": [1, 2, 4]}),
    "MLP": (MLPClassifier(max_iter=1000, random_state=0),
            {"hidden_layer_sizes": [(64, 64), (128, 128), (256, 256, 128)], "learning_rate_init": [1e-4, 1e-3, 1e-2]}),
    "XGBoost": (XGBClassifier(n_jobs=1, random_state=0, tree_method="hist", eval_metric="logloss"),
                {"n_estimators": [100, 300, 500], "max_depth": [3, 6, 9], "learning_rate": [0.01, 0.05, 0.1, 0.2],
                 "subsample": [0.7, 0.9, 1.0]}),
}
ORDER = ["Logistic Regression", "Linear SVM", "RBF SVM", "Random Forest", "MLP", "XGBoost"]


def fit_predict(name, Xtr, ytr, Xte, seed):
    y01 = (ytr + 1) // 2
    if name == "Logistic Regression":
        m = LogisticRegression(C=1.0, penalty="l2", max_iter=5000).fit(Xtr, y01); best = {}
    elif name == "Linear SVM":
        m = SVC(kernel="linear", C=1.0).fit(Xtr, y01); best = {}
    else:
        est, grid = GRIDS[name]
        gs = GridSearchCV(est, grid, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          scoring="accuracy", n_jobs=N_JOBS, refit=True).fit(Xtr, y01)
        m, best = gs.best_estimator_, gs.best_params_
    return 2 * m.predict(Xte) - 1, best


rows = []
t_all = time.perf_counter()
_csv = OUT / ("classical_matched_trials.csv" if MODE == "matched" else "classical_nested_cv.csv")
if _csv.exists():                     # resume: keep completed units, skip them below
    rows = pd.read_csv(_csv).to_dict("records")
done_m = {(r["dataset"], r["seed"]) for r in rows} if MODE == "matched" else set()
done_n = {(r["dataset"], r["repetition"]) for r in rows} if MODE == "nested" else set()
if MODE == "matched":
    for ds in C.DATASETS:
        for seed in C.SEEDS:
            if (ds, seed) in done_m and sum(1 for r in rows if r["dataset"] == ds and r["seed"] == seed) == 12:
                continue
            rows = [r for r in rows if not (r["dataset"] == ds and r["seed"] == seed)]
            for dim in (4, 8):
                Ftr, Fte, ytr, yte = C.matched_data(ds, seed, dim)
                for name in ORDER:
                    t0 = time.perf_counter()
                    pred, best = fit_predict(name, Ftr, ytr, Fte, seed)
                    rows.append(dict(dataset=ds, model=name, dimension=dim, seed=seed,
                                     test_accuracy=float(np.mean(pred == yte)), n_test=len(yte),
                                     fit_seconds=time.perf_counter() - t0, best_params=str(best)))
                print(f"{ds} seed {seed} dim {dim}: " + ", ".join(f"{r['model'][:6]}={r['test_accuracy']:.3f}" for r in rows[-6:])
                      + f"  [{time.perf_counter()-t_all:.0f}s]", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "classical_matched_trials.csv", index=False)
elif MODE == "nested":
    for ds in C.DATASETS:
        X, y = C.load(ds)
        for rep in range(5):
            if (ds, rep) in done_n and sum(1 for r in rows if r["dataset"] == ds and r["repetition"] == rep) == 30:
                continue
            rows = [r for r in rows if not (r["dataset"] == ds and r["repetition"] == rep)]
            outer = StratifiedKFold(5, shuffle=True, random_state=rep)
            for fold, (tr, te) in enumerate(outer.split(X, y)):
                sc = MinMaxScaler().fit(X[tr])
                Xtr, Xte = sc.transform(X[tr]), np.clip(sc.transform(X[te]), 0, 1)
                for name in ORDER:
                    pred, best = fit_predict(name, Xtr, y[tr], Xte, 100 * rep + fold)
                    rows.append(dict(dataset=ds, model=name, repetition=rep, fold=fold,
                                     test_accuracy=float(np.mean(pred == y[te])), best_params=str(best)))
                print(f"{ds} rep {rep} fold {fold}: " + ", ".join(f"{r['model'][:6]}={r['test_accuracy']:.3f}" for r in rows[-6:])
                      + f"  [{time.perf_counter()-t_all:.0f}s]", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "classical_nested_cv.csv", index=False)
print("done", time.perf_counter() - t_all)
