#!/usr/bin/env bash
# Full pipeline. Each step writes to results/; run_classical.py resumes if interrupted.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # avoid oversubscription; parallelism via N_JOBS
N_JOBS=${N_JOBS:-2}                               # CPU cores for grid search

python test_vqc_core.py                            # 0. validate simulator against Aer density matrix (~1 min)
python -u run_vqc.py results          | tee log_vqc.txt        # 1. primary VQC, Tables 7-8 (~1 h on 2 cores)
python -u run_classical.py matched results $N_JOBS | tee log_matched.txt  # 2. Table 6 (~1.3 h)
python -u run_classical.py nested  results $N_JOBS | tee log_nested.txt   # 3. Table 5 (~1.5 h)
python -u run_aer.py nairobi results  | tee log_nairobi.txt    # 4. FakeNairobiV2, Table 10 / Fig 9
python -u run_aer.py timing  results  | tee log_timing.txt     # 5. Aer timing, Table 9 / Fig 8 (run on an idle machine)
MPLBACKEND=Agg python build_results.py results tables figures   # 6. all tables, paired tests, Figs 6-9, summary.json
MPLBACKEND=Agg python training_diagnostics.py results tables  # 6b. optimisation diagnostics (training_diagnostics.json)
MPLBACKEND=Agg python make_analytical_figures.py figures_analytical           # 7. Figs 4, 5 (analytical)
echo "Done. Tables in tables/, figures in figures/, numbers in tables/summary.json"
