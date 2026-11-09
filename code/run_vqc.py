"""Primary VQC experiment: 2 datasets x 3 configurations x 15 splits.

For every (dataset, n, L, seed):
  * retrained protocol: Algorithm 1 is run separately at eps_2q in {1e-3, 1e-2, 1e-1}
    and the trained model is evaluated on the test partition at the same noise level;
  * fixed-model protocol: the parameters trained at eps_2q = 1e-2 are re-evaluated,
    without retraining, at all three eps_2q values.
eps_1q = 1e-3 throughout. Output: results/vqc_trials.csv (one row per evaluation) and
results/vqc_loss_history.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import vqc_core as V

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT.mkdir(parents=True, exist_ok=True)

rows, hist_rows = [], []
t_all = time.perf_counter()
for ds in C.DATASETS:
    for n, L in C.CONFIGS:
        for seed in C.SEEDS:
            Ftr, Fte, ytr, yte = C.matched_data(ds, seed, n)
            trained = {}
            for e2 in C.EPS_2Q_GRID:
                t0 = time.perf_counter()
                theta, rng, hist = V.spsa_train(Ftr, ytr, n, L, C.EPS_1Q, e2, seed,
                                                n_iter=C.N_ITER, S=C.SHOTS, history=True)
                t_train = time.perf_counter() - t0
                acc_tr, _ = V.evaluate(theta, Ftr, ytr, n, L, C.EPS_1Q, e2, rng, C.SHOTS)
                t0 = time.perf_counter()
                acc_te, m = V.evaluate(theta, Fte, yte, n, L, C.EPS_1Q, e2, rng, C.SHOTS)
                t_eval = time.perf_counter() - t0
                acc_exact = float(np.mean(V.predict_labels(m) == yte))
                trained[e2] = theta
                rows.append(dict(dataset=ds, n_qubits=n, n_layers=L, seed=seed, protocol="retrained",
                                 eps_2q_train=e2, eps_2q=e2, test_accuracy=acc_te,
                                 test_accuracy_noiseless_readout=acc_exact, train_accuracy=acc_tr,
                                 n_train=len(ytr), n_test=len(yte), train_seconds=t_train,
                                 eval_seconds=t_eval))
                for k, l in enumerate(hist):
                    hist_rows.append(dict(dataset=ds, n_qubits=n, n_layers=L, seed=seed, eps_2q=e2,
                                          iteration=k, loss=l))
            theta = trained[C.EPS_2Q_PRIMARY]
            for j, e2 in enumerate(C.EPS_2Q_GRID):
                rng = np.random.default_rng([seed, 7, j])
                acc_te, _ = V.evaluate(theta, Fte, yte, n, L, C.EPS_1Q, e2, rng, C.SHOTS)
                rows.append(dict(dataset=ds, n_qubits=n, n_layers=L, seed=seed, protocol="fixed_model",
                                 eps_2q_train=C.EPS_2Q_PRIMARY, eps_2q=e2, test_accuracy=acc_te,
                                 n_train=len(ytr), n_test=len(yte)))
            r = [x for x in rows if x["protocol"] == "retrained" and x["eps_2q"] == 1e-2][-1]
            print(f"{ds:13s} {n}Q-{L}L seed {seed}: test acc @1e-2 = {r['test_accuracy']:.4f} "
                  f"(train {r['train_accuracy']:.3f}, {r['train_seconds']:.1f}s)  [{time.perf_counter()-t_all:.0f}s]",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT / "vqc_trials.csv", index=False)
        pd.DataFrame(hist_rows).to_csv(OUT / "vqc_loss_history.csv", index=False)
print("done", time.perf_counter() - t_all)
