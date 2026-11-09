# Response to the six revision points (with regenerated results)

`main_diff.pdf` marks every change against the version you sent. Table numbers below
refer to the **revised** manuscript. A new paired-test table (Table 8) shifts the later
tables up by one.

**Status:** all six points are done. The original trial-level CSVs, paired-tests file
and Pauli-propagation code could not be found: they are not in the Overleaf bundle, the
public repository, or the two circulated notebooks. So the experiments were
**re-implemented from the protocol in the manuscript and rerun in full**. Every accuracy,
timing and calibrated-model value in Sections 7–12 now comes from these new runs. The
values you quoted (0.5053 vs 0.5158, etc.) are superseded, and the earlier "blocked"
notes no longer apply.

## How the results were produced

* **Code** (`code/`): it implements Algorithm 1 exactly as written. The noisy
  ⟨Z₀⟩ comes from Heisenberg-picture Pauli propagation; it matches Qiskit Aer
  density-matrix simulation to 5×10⁻¹⁵ on all three configurations.
* **Experiments:**
  * matched classical baselines with inner-CV grid search on the same 15 splits and
    PCA features;
  * 5×5 nested CV;
  * FakeNairobiV2 on all 15 splits;
  * Aer timing on the transpiled final circuit.
* **Runs:**
  * The full pipeline ran on Kaggle (4-vCPU Xeon 2.00 GHz, Python 3.11, pinned
    versions).
  * A second, independent run on another machine gave **identical** test accuracies
    for all 540 VQC and 360 classical records.
  * All tables and Figures 6–9 are generated from the CSVs by `build_results.py`.
  * I cross-checked every number against the raw CSVs with a separate script that
    shares no code with the build script.

## 1. Figure 3 and analytical Figures 4, 5, 8 — done
Figure 3 is unchanged from the previous round: the quantikz redraw with the final
ansatz. Figures 4 and 5 are unchanged: analytical, G₁q = n(2L+2), G₂q = (n−1)L,
C = 36/76/136. **Figure 8** now plots the newly measured timings against C.

## 2. Tables 6/7 vs CSV, A_ref, Figure 6, Table 12 — done (new values)
| | Old | New |
|---|---|---|
| A_ref, Breast Cancer | 0.9678 (Linear SVM 8D) | **0.9708 (MLP 8D)** |
| A_ref, digits | 1.0000 | 1.0000 (RBF SVM 4D and three 8D models) |
| Best VQC, BC / digits | 0.6731 / 0.6556 | **0.7409 (4Q–2L) / 0.7222 (8Q–2L)** |
| Max VQC CI upper / min classical CI lower | 0.7917 / 0.9111 | 0.8509 / 0.9251 |
| Pre-VDCS 4Q–2L / 8Q–2L / 8Q–4L | 87.35 / 45.62 / 38.82 | **89.74 / 64.09 / 53.79** (same ranking) |
| ΔA (Table 13), BC / digits | −0.4625 / −0.3444 | −0.4860 / −0.4583 |

* **Bootstrap:** B = 10,000, seed 2026, now stated in Section 5.
* **Figure 6:** now also shows the 15 split-level points and the majority-class rate.

## 3. Pauli propagation vs Aer vs FakeNairobiV2 — done
Section 5 defines three pathways and ties every result to one of them.

* **Primary simulation:** depolarizing channel after every gate, including encoding,
  with the Qiskit convention (non-identity Paulis scaled by 1−p).
* **Aer:** used for timing and for FakeNairobiV2. The two questions from last round
  are now answered by the implementation itself.

## 4. Split-level statements — done (option A)
All six "not retained" statements are replaced. Trial-level outcomes are released, so
the comparison is now paired. The new **Table 8** gives, for each configuration:

* the mean paired difference with its bootstrap CI;
* the number of splits where the VQC wins;
* Holm-adjusted p-values from the paired t-test and the Wilcoxon test.

All six configurations are significantly below their reference: Holm-adjusted
p ≤ 0.0017 (t) and ≤ 0.0015 (Wilcoxon). The VQC wins 1 of 90 split comparisons.

## 5. Digits dataset — done (unchanged from last round)

## 6. Provenance limits — done
Timing, FakeNairobiV2 and the noise table are now regenerated, so their provenance
caveats have been removed. Two items still carry provenance caveats in the
Reproducibility Statement and Section S2:

* the supplementary diagnostics (Tables S3, S5–S7);
* the scheduling simulation.

Table S2 is analytical and is fine.

Section S2 now also states that these legacy diagnostics use a different protocol from the main results. They used Adam, SGD or COBYLA for about 60–85 iterations, with PCA at 70% variance. Their VQC accuracies (0.72–0.79) are therefore not comparable with Table 7. The closing sentence that ranked noise as the largest sensitivity factor has been replaced, because the new 15-split analysis shows noise has little effect.

---

## Results that change the paper's story — please read

1. **The VQC is barely trained under Algorithm 1 as specified.**
   * **Small steps:** with a = 0.05 and A = 10, a_k goes from 0.0118 to 0.0065.
   * **Little loss reduction:** over 20 iterations, the loss falls by only 16% on
     average.
   * **Initialization decides the outcome:** train and test accuracy correlate at
     r = 0.992 across trials.
   * **Inverted predictions:** 35/90 trials score below 0.5, a sign-inverted rule
     inherited from the random θ.
   * **Below a trivial classifier:** the two 8-qubit Breast Cancer means (0.447,
     0.485) are below the majority-class rate (0.632).

   I state this openly in Section 7, Threats (new "Optimization budget" item), and the
   Discussion. **A reviewer will very likely ask for a converged baseline.** I
   recommend a short sensitivity study: e.g. 200 iterations, and/or tuned gains,
   and/or 3 restarts. The code needs one parameter change, and it takes about 1 h on
   Kaggle. Algorithm 1 itself stays as is.
2. **Noise barely matters at this budget.**
   * **Fixed model:** accuracy changes by ≤ 0.006 across ε₂q = 10⁻³…10⁻¹, because the
     sign of ⟨Z₀⟩ rarely flips.
   * **Retrained:** accuracy changes by 0.016–0.061, inside the split-to-split SD.

   Table 9 is now a 15-split fixed-vs-retrained table, replacing the old single-run
   table.
3. **Timing is reversed.** On the final circuit, *compilation* dominates local Aer
   time: execution share is 8.3–11.1%, not 40–98%.
   * Transpilation adds no SWAPs: CX = 6/14/28, depth 14/18/30.
   * The old "execution dominates at 8 qubits" claim has been removed throughout.
4. **FakeNairobiV2 now uses 15 splits** instead of 3.
   * **Accuracy:** mean 0.722 ± 0.169, within 0.02 of the synthetic model on the same
     splits (Wilcoxon p = 0.09).
   * **Physical layout:** [4, 5, 3, 1].

The abstract, claim–evidence table, interpretation and conclusion are rewritten to
match. Algorithm 1, Eqs. 2–5, C₀ = 36 and the Reproducibility Statement structure are
kept.

## Before submission
* **Push the code and data:** add `code/` (with `results/`, `tables/`, `figures/`) to
  github.com/AnuragB2004/VQC-NISQ. The Reproducibility Statement now says the
  repository contains them.
* **Decide on the convergence study** described in item 1 above.

## Files
* `main.tex`, `supplementary.tex`, `main.pdf` (19 pp.), `main_diff.pdf` (tracked changes; table bodies are shown in their new form without cell-by-cell markup, because latexdiff markup breaks inside these tables).
* Figures 6–9 are regenerated under their Overleaf filenames.
* `code/` contains everything needed to reproduce the results:
  * the source modules and run scripts;
  * `run_all.sh` and the Kaggle notebook;
  * `results/` (trial CSVs and run logs), `tables/` and `figures/`.
