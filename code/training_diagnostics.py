"""Optimisation diagnostics for the primary VQC run (eps_2q = 1e-2, retrained protocol).

Reports, per configuration and pooled:
  * relative reduction of the SPSA loss estimate between iteration 1 and iteration 20;
  * Pearson correlation between train and test accuracy across trials;
  * number of trials with test accuracy below 0.5 and below the majority-class rate;
  * mean test accuracy minus majority-class rate.
Usage: python training_diagnostics.py results tables
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "tables"); OUT.mkdir(parents=True, exist_ok=True)

h = pd.read_csv(RES / "vqc_loss_history.csv"); h = h[np.isclose(h.eps_2q, 1e-2)]
v = pd.read_csv(RES / "vqc_trials.csv"); r = v[(v.protocol == "retrained") & np.isclose(v.eps_2q, 1e-2)]
keys = ["dataset", "n_qubits", "n_layers", "seed"]
g = h.sort_values("iteration").groupby(keys).loss
drop = ((g.first() - g.last()) / g.first()).rename("rel_drop").reset_index()
r = r.merge(drop, on=keys)


def majority(ds, seed):
    _, _, ytr, yte = C.matched_data(ds, seed, 4)
    maj = 1 if np.sum(ytr == 1) >= np.sum(ytr == -1) else -1
    return float(np.mean(yte == maj))


r["majority"] = [majority(d, s) for d, s in zip(r.dataset, r.seed)]
out = {"per_config": {}}
for (ds, n, L), x in r.groupby(["dataset", "n_qubits", "n_layers"]):
    out["per_config"][f"{ds}|{n}|{L}"] = dict(
        rel_loss_drop_mean=float(x.rel_drop.mean()), rel_loss_drop_min=float(x.rel_drop.min()),
        rel_loss_drop_max=float(x.rel_drop.max()),
        r_train_test=float(np.corrcoef(x.train_accuracy, x.test_accuracy)[0, 1]),
        n_below_half=int((x.test_accuracy < 0.5).sum()),
        n_below_majority=int((x.test_accuracy < x.majority).sum()),
        mean_minus_majority=float((x.test_accuracy - x.majority).mean()))
out["pooled"] = dict(
    n_trials=int(len(r)),
    rel_loss_drop_mean=float(r.rel_drop.mean()), rel_loss_drop_median=float(r.rel_drop.median()),
    r_train_test=float(np.corrcoef(r.train_accuracy, r.test_accuracy)[0, 1]),
    n_below_half=int((r.test_accuracy < 0.5).sum()),
    n_below_majority=int((r.test_accuracy < r.majority).sum()))
(OUT / "training_diagnostics.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
