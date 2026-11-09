# VQC system-level evaluation — experiment code

Reimplementation of the full experimental protocol in the revised manuscript.

## Setup
```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./run_all.sh            # or run the steps in run_all.sh one at a time
```
Set `N_JOBS=<cores>` to speed up the classical grid searches (e.g. `N_JOBS=8 ./run_all.sh`).

## Files
| File | Purpose | Output (results/) | Manuscript |
|---|---|---|---|
| `vqc_core.py` | Pauli-propagation simulator (Heisenberg picture, Pauli transfer matrices), Qiskit depolarizing convention, binomial shot sampling (S=1024), SPSA (Algorithm 1: a=0.05, c=0.10, A=10, α=0.602, γ=0.101, 20 iterations) | — | Sec. 5 |
| `test_vqc_core.py` | Checks ⟨Z₀⟩ against a Qiskit Aer density-matrix simulation (agreement ~1e-14) | — | pathway (i) |
| `common.py` | Datasets (sklearn Breast Cancer; sklearn Digits 0 vs 1), 15 stratified 80/20 splits (seeds 42–56), MinMax→PCA(d)→MinMax fitted on train only | — | Sec. 5 |
| `run_vqc.py` | 2 datasets × {4Q-2L, 8Q-2L, 8Q-4L} × 15 splits; retrained at ε₂q ∈ {1e-3,1e-2,1e-1} plus fixed-model re-evaluation; ε₁q = 1e-3 | `vqc_trials.csv`, `vqc_loss_history.csv` | Tables 7, 8; Fig 7 |
| `run_classical.py matched` | LR, linear SVM, and grid-searched RBF SVM / RF / MLP / XGBoost (inner 5-fold CV) on the same splits and PCA features (d = 4, 8) | `classical_matched_trials.csv` | Table 6; Fig 6 |
| `run_classical.py nested` | 5 × 5 nested CV on full min-max features | `classical_nested_cv.csv` | Table 5 |
| `run_aer.py nairobi` | 4Q-2L Breast Cancer, trained and tested with AerSimulator.from_backend(FakeNairobiV2), 15 splits | `fakenairobi_trials.csv` | Table 10; Fig 9 |
| `run_aer.py timing` | Transpiled final circuit (basis rz/sx/x/cx, linear map, opt. level 2, SABRE), Aer depolarizing noise, 1 warm-up + 10 timed reps | `aer_timing.csv`, `aer_timing_meta.json` | Table 9; Fig 8 |
| `training_diagnostics.py` | Loss reduction over the 20 SPSA iterations, train/test accuracy correlation, trials below 0.5 and below majority-class rate | `tables/training_diagnostics.json` | Sec. 7, Threats |
| `build_results.py` | Mean, SD (ddof=1), bootstrap 95% CI (B=10,000, seed 2026), paired t / Wilcoxon / Holm tests, A_ref, Pre-VDCS scores, break-even, Figs 6–9 | `tables/*.tex`, `tables/summary.json`, `figures/*.png` | Tables 5–12, paired-test table |
| `make_analytical_figures.py` | G₁q = n(2L+2), G₂q = (n−1)L, C = 36/76/136 | Figs 4, 5 | Sec. 4 |

`build_results.py` skips the timing and Nairobi tables if those CSVs don't exist yet, so you can build partial results at any point.

## Results included (`results/`, `tables/`, `figures/`)
The complete run used for the revised manuscript. It was executed on Kaggle (4-vCPU Intel Xeon 2.00 GHz, Python 3.11.18, pinned versions above). `results/log_*.txt` are its console logs. An independent second run on a 2-vCPU machine produced identical test accuracies for all 540 VQC and 360 matched classical records. Rerunning reproduces every accuracy exactly (all RNGs are seeded); only `*_seconds` columns and Aer timings depend on the hardware.

`tables/` and `figures/` were built from `results/` with `build_results.py` and `training_diagnostics.py`. On Kaggle, run the build step with `MPLBACKEND=Agg` (the notebook now sets this), otherwise matplotlib in the separate environment fails to load Kaggle's inline backend.

## Notes
- With 20 SPSA iterations (Algorithm 1 gains) the VQC is training-limited: the loss estimate falls by 16% on average, train and test accuracy correlate at r = 0.99 across trials, and 35 of 90 trials score below 0.5 (a sign-inverted decision rule inherited from the random initialization). The two Breast Cancer 8-qubit means are below the majority-class rate (0.632). The manuscript reports this as a limitation of the stated budget.
- Run the timing step on an otherwise idle machine; the timings are hardware-specific (report the CPU).

## Table files vs revised manuscript numbering
`table5_nested` → Table 5, `table6_matched` → Table 6, `table7_vqc` → Table 7, `table_paired` → Table 8,
`table8_noise` → Table 9, `table9_timing` → Table 10, Nairobi → Table 11 (from `summary.json`),
`table11_prevdcs` → Table 12, `table12_breakeven` → Table 13.
