"""Build every reported table and data figure from the trial-level CSVs in results/.

Statistics: mean; sample SD (ddof = 1); percentile bootstrap 95% CI of the mean
(B = 10,000 resamples, numpy default_rng(2026)). Paired comparisons (same split seed):
mean paired difference with percentile bootstrap CI, paired t-test and exact Wilcoxon
signed-rank test, Holm correction over the six VQC-vs-reference comparisons.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "tables"); OUT.mkdir(parents=True, exist_ok=True)
FIG = Path(sys.argv[3] if len(sys.argv) > 3 else "figures"); FIG.mkdir(parents=True, exist_ok=True)
B, BOOT_SEED = 10_000, 2026
rng = np.random.default_rng(BOOT_SEED)
ORDER = ["Logistic Regression", "Linear SVM", "RBF SVM", "Random Forest", "MLP", "XGBoost"]
COL = {(4, 2): "#2a78d6", (8, 2): "#eb6834", (8, 4): "#1baf7a"}
MK = {(4, 2): "o", (8, 2): "s", (8, 4): "D"}
TAG = {c: f"{c[0]}Q\u2013{c[1]}L" for c in C.CONFIGS}
plt.rcParams.update({"font.size": 10, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": "#dddddd", "grid.linewidth": 0.6})


def g1q(n, L): return n * (2 * L + 2)
def g2q(n, L): return (n - 1) * L
def burden(n, L): return g1q(n, L) + 2 * g2q(n, L)
def fid(n, L, e2=1e-2): return (1 - C.EPS_1Q) ** g1q(n, L) * (1 - e2) ** g2q(n, L)


def boot(x):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summ(x):
    x = np.asarray(x, float); lo, hi = boot(x)
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)), lo=lo, hi=hi)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


S = {"bootstrap": dict(B=B, seed=BOOT_SEED)}
vqc = pd.read_csv(RES / "vqc_trials.csv")
cls = pd.read_csv(RES / "classical_matched_trials.csv")
ret = vqc[vqc.protocol == "retrained"]; fix = vqc[vqc.protocol == "fixed_model"]

# integrity: identical seeds everywhere
for ds in C.DATASETS:
    for (n, L) in C.CONFIGS:
        s = set(ret[(ret.dataset == ds) & (ret.n_qubits == n) & (ret.n_layers == L)].seed)
        assert s == set(C.SEEDS), (ds, n, L, s)
    for (m, d), g in cls[cls.dataset == ds].groupby(["model", "dimension"]):
        assert set(g.seed) == set(C.SEEDS), (ds, m, d)

# ---------------------------------------------------------------- majority-class reference
S["majority"] = {}
for ds in C.DATASETS:
    acc = []
    for seed in C.SEEDS:
        _, _, ytr, yte = C.matched_data(ds, seed, 4)
        maj = 1 if (ytr == 1).sum() >= (ytr == -1).sum() else -1
        acc.append(float(np.mean(yte == maj)))
    S["majority"][ds] = summ(acc)

# ---------------------------------------------------------------- Table 7
S["vqc"] = {}; rows = []
for ds in C.DATASETS:
    for n, L in C.CONFIGS:
        x = ret[(ret.dataset == ds) & (ret.n_qubits == n) & (ret.n_layers == L) & np.isclose(ret.eps_2q, 1e-2)].sort_values("seed")
        s = summ(x.test_accuracy); s["train_mean"] = float(x.train_accuracy.mean())
        s["min"], s["max"] = float(x.test_accuracy.min()), float(x.test_accuracy.max())
        s["below_half"] = int((x.test_accuracy < 0.5).sum())
        S["vqc"][f"{ds}|{n}|{L}"] = s
        rows.append(f"{C.LABEL[ds]} & {n} & {L} & {s['mean']:.4f} & {s['sd']:.4f} & [{s['lo']:.4f}, {s['hi']:.4f}] \\\\")
(OUT / "table7_vqc.tex").write_text("\n".join(rows) + "\n")

# ---------------------------------------------------------------- Table 6 + A_ref
S["classical"] = {}; S["A_ref"] = {}; blocks = []
for ds in C.DATASETS:
    sub = cls[cls.dataset == ds]; lines = []; best = (None, -1)
    for d in (4, 8):
        for m in ORDER:
            s = summ(sub[(sub.model == m) & (sub.dimension == d)].sort_values("seed").test_accuracy)
            S["classical"][f"{ds}|{m}|{d}"] = s
            lines.append(f"& {m:19s} & {d} & {s['mean']:.4f} & {s['sd']:.4f} & [{s['lo']:.4f}, {s['hi']:.4f}] \\\\")
            if s["mean"] > best[1] + 1e-12: best = ((m, d), s["mean"])
    S["A_ref"][ds] = dict(model=best[0][0], dimension=best[0][1], mean=best[1])
    blocks.append(f"\\multirow{{12}}{{*}}{{{C.LABEL[ds]}}}\n" + "\n".join(lines))
(OUT / "table6_matched.tex").write_text("\n\n\\midrule\n\n".join(blocks) + "\n")
S["vqc_ci_max_upper"] = max(v["hi"] for v in S["vqc"].values())
S["classical_ci_min_lower"] = min(v["lo"] for v in S["classical"].values())

# ---------------------------------------------------------------- paired tests
pr = []
for ds in C.DATASETS:
    R = S["A_ref"][ds]
    ref = cls[(cls.dataset == ds) & (cls.model == R["model"]) & (cls.dimension == R["dimension"])].set_index("seed").test_accuracy
    for n, L in C.CONFIGS:
        # also the best classical model at the VQC's own dimension
        v = ret[(ret.dataset == ds) & (ret.n_qubits == n) & (ret.n_layers == L) & np.isclose(ret.eps_2q, 1e-2)].set_index("seed").test_accuracy
        d = (v.loc[C.SEEDS] - ref.loc[C.SEEDS]).to_numpy()
        lo, hi = boot(d)
        t = stats.ttest_rel(v.loc[C.SEEDS], ref.loc[C.SEEDS])
        w = stats.wilcoxon(d, zero_method="wilcox", method="exact" if np.all(d != 0) else "approx")
        pr.append(dict(dataset=ds, n=n, L=L, reference=f"{R['model']} ({R['dimension']}D)", mean_diff=float(d.mean()),
                       lo=lo, hi=hi, wins=int((d > 0).sum()), t=float(t.statistic), p_t=float(t.pvalue),
                       W=float(w.statistic), p_w=float(w.pvalue)))
pt = pd.DataFrame(pr); pt["p_t_holm"] = holm(pt.p_t); pt["p_w_holm"] = holm(pt.p_w)
pt.to_csv(OUT / "paired_tests.csv", index=False); S["paired"] = pt.to_dict("records")


def fmt_p(p):
    return f"{p:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}" if p < 1e-3 else f"{p:.3f}"


(OUT / "table_paired.tex").write_text("\n".join(
    f"{C.LABEL[r.dataset]} & {r.n} & {r.L} & {r.reference} & ${r.mean_diff:.4f}$ & [${r.lo:.4f}$, ${r.hi:.4f}$] & {r.wins}/15 & ${fmt_p(r.p_t_holm)}$ & ${fmt_p(r.p_w_holm)}$ \\\\"
    for r in pt.itertuples()) + "\n")

# ---------------------------------------------------------------- Table 8 / Figure 7 (noise sensitivity)
S["noise"] = {}; rows = []
for ds in C.DATASETS:
    for n, L in C.CONFIGS:
        cells = []
        for proto, df in (("fixed", fix), ("retrained", ret)):
            for e in C.EPS_2Q_GRID:
                x = df[(df.dataset == ds) & (df.n_qubits == n) & (df.n_layers == L) & np.isclose(df.eps_2q, e)].test_accuracy
                s = summ(x); S["noise"][f"{proto}|{ds}|{n}|{L}|{e:g}"] = s
                cells.append(f"{s['mean']:.3f} $\\pm$ {s['sd']:.3f}")
        rows.append(f"{C.LABEL[ds]} & {n} & {L} & " + " & ".join(cells) + " \\\\")
(OUT / "table8_noise.tex").write_text("\n".join(rows) + "\n")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw=dict(width_ratios=[1.15, 1]))
lab = [f"{'BC' if ds == 'breast_cancer' else 'Digits'} {TAG[c]}" for ds in C.DATASETS for c in C.CONFIGS]
M = np.array([[S["noise"][f"fixed|{ds}|{c[0]}|{c[1]}|{e:g}"]["mean"] for e in C.EPS_2Q_GRID] for ds in C.DATASETS for c in C.CONFIGS])
im = ax[0].imshow(M, cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
for i in range(M.shape[0]):
    for j in range(3):
        ax[0].text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=9, color="white" if M[i, j] > 0.75 else "#222222")
ax[0].set_xticks(range(3), ["$10^{-3}$", "$10^{-2}$", "$10^{-1}$"]); ax[0].set_yticks(range(6), lab)
ax[0].set_xlabel("Evaluation $\\epsilon_{2q}$ (parameters trained at $10^{-2}$)")
ax[0].set_title("(a) Fixed-model mean test accuracy (15 splits)")
for s in ax[0].spines.values(): s.set_visible(False)
fig.colorbar(im, ax=ax[0], fraction=0.04, pad=0.02)
amp_f = M.max(1) - M.min(1)
Mr = np.array([[S["noise"][f"retrained|{ds}|{c[0]}|{c[1]}|{e:g}"]["mean"] for e in C.EPS_2Q_GRID] for ds in C.DATASETS for c in C.CONFIGS])
amp_r = Mr.max(1) - Mr.min(1)
y = np.arange(6)
ax[1].barh(y - 0.19, amp_f, 0.36, color="#2a78d6", label="Fixed model")
ax[1].barh(y + 0.19, amp_r, 0.36, color="#eb6834", label="Retrained per $\\epsilon_{2q}$")
for i in range(6):
    ax[1].text(amp_f[i] + 0.002, i - 0.19, f"{amp_f[i]:.3f}", va="center", fontsize=8.5)
    ax[1].text(amp_r[i] + 0.002, i + 0.19, f"{amp_r[i]:.3f}", va="center", fontsize=8.5)
ax[1].set_yticks(y, lab); ax[1].invert_yaxis(); ax[1].set_xlabel("Range of mean accuracy across $\\epsilon_{2q}$")
ax[1].set_title("(b) Sensitivity amplitude"); ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2); ax[1].grid(axis="x")
ax[1].set_xlim(0, max(amp_f.max(), amp_r.max()) * 1.25 + 0.01)
fig.tight_layout(); fig.savefig(FIG / "Figure_7_Noise_Sensitivity.png", bbox_inches="tight"); plt.close(fig)
S["noise_amplitude"] = dict(fixed=dict(zip(lab, amp_f.tolist())), retrained=dict(zip(lab, amp_r.tolist())))

# ---------------------------------------------------------------- Figure 6
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for k, (axx, ds) in enumerate(zip(axes, C.DATASETS)):
    R = S["A_ref"][ds]
    for i, c in enumerate(C.CONFIGS):
        x = ret[(ret.dataset == ds) & (ret.n_qubits == c[0]) & (ret.n_layers == c[1]) & np.isclose(ret.eps_2q, 1e-2)].test_accuracy
        jit = np.random.default_rng(i).uniform(-0.12, 0.12, len(x))
        axx.scatter(i + jit, x, s=14, color=COL[c], alpha=0.35, linewidths=0, zorder=2)
        s = S["vqc"][f"{ds}|{c[0]}|{c[1]}"]
        axx.errorbar(i, s["mean"], yerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]], fmt=MK[c], color=COL[c],
                     ms=9, capsize=6, lw=2, mec="white", mew=1.2, zorder=3)
        axx.text(i, 0.03, f"$\\Delta A$={s['mean'] - R['mean']:.3f}", ha="center", va="bottom", fontsize=9, color="#222222")
    axx.axhline(R["mean"], ls="--", color="#555555", lw=1.4)
    axx.text(-0.45, R["mean"] + 0.012, f"Strongest matched classical: {R['model']} ({R['dimension']}D) = {R['mean']:.4f}", ha="left", va="bottom", fontsize=8.5, color="#333333")
    M = S["majority"][ds]["mean"]
    axx.axhline(M, ls=":", color="#888888", lw=1.2, zorder=1)
    axx.text(0.5, M + 0.01, f"Majority class = {M:.3f}", ha="center", va="bottom", fontsize=8, color="#555555",
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85), zorder=4)
    axx.set_xticks(range(3), [TAG[c] for c in C.CONFIGS]); axx.set_xlim(-0.5, 2.5); axx.set_ylim(0, 1.12)
    axx.grid(axis="y"); axx.set_xlabel("VQC configuration"); axx.set_title(f"({'ab'[k]}) {C.LABEL[ds]}")
axes[0].set_ylabel("Test accuracy ($\\epsilon_{2q}=10^{-2}$, 15 matched splits)")
fig.tight_layout(); fig.savefig(FIG / "Figure_6_Performance_Landscape.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Table 5 (nested CV)
nest = pd.read_csv(RES / "classical_nested_cv.csv") if (RES / "classical_nested_cv.csv").exists() else None
S["nested"] = {}
if nest is not None:
    rows = []
    for ds in C.DATASETS:
        for j, m in enumerate(ORDER):
            g = nest[(nest.dataset == ds) & (nest.model == m)]
            rep_means = g.groupby("repetition").test_accuracy.mean()
            S["nested"][f"{ds}|{m}"] = dict(mean=float(g.test_accuracy.mean()), sd_reps=float(rep_means.std(ddof=1)),
                                            sd_folds=float(g.test_accuracy.std(ddof=1)), n_folds=len(g))
            pre = f"\\multirow{{6}}{{*}}{{{C.LABEL[ds]}}} " if j == 0 else ""
            rows.append(f"{pre}& {m} & {g.test_accuracy.mean():.4f} & {rep_means.std(ddof=1):.4f} \\\\")
        rows.append("\\midrule")
    (OUT / "table5_nested.tex").write_text("\n".join(rows[:-1]) + "\n")

# ---------------------------------------------------------------- Table 9 / Figure 8 (timing)
T = {}
if (RES / "aer_timing.csv").exists():
    tm = pd.read_csv(RES / "aer_timing.csv"); meta = json.loads((RES / "aer_timing_meta.json").read_text())
    rows = []
    for n, L in C.CONFIGS:
        g = tm[(tm.n_qubits == n) & (tm.n_layers == L)]
        tot = g.compile_s + g.execute_s; share = 100 * g.execute_s / tot
        T[(n, L)] = float(tot.mean())
        k = f"{n}Q-{L}L"
        S.setdefault("timing", {})[k] = dict(compile=float(g.compile_s.mean()), compile_sd=float(g.compile_s.std(ddof=1)),
                                             execute=float(g.execute_s.mean()), execute_sd=float(g.execute_s.std(ddof=1)),
                                             total=float(tot.mean()), share=float(share.mean()), **meta[k])
        t = S["timing"][k]
        rows.append(f"{n} & {L} & {t['compile']:.3f} $\\pm$ {t['compile_sd']:.3f} & {t['execute']:.3f} $\\pm$ {t['execute_sd']:.3f} & {t['share']:.1f} & {t['transpiled_depth_mean']:.1f} & {t['transpiled_cx_mean']:.0f} \\\\")
    (OUT / "table9_timing.tex").write_text("\n".join(rows) + "\n")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for c in C.CONFIGS:
        t = S["timing"][f"{c[0]}Q-{c[1]}L"]
        ax.scatter(burden(*c), t["total"], s=4 * burden(*c), color=COL[c], marker=MK[c], edgecolor="white", linewidth=1.5, zorder=3)
        ax.errorbar(burden(*c), t["total"], yerr=np.hypot(t["compile_sd"], t["execute_sd"]), color="#555555", lw=1, capsize=3, zorder=2)
        ax.annotate(f"{TAG[c]} (C={burden(*c)})\n{t['total']:.3f} s, {t['share']:.1f}% execution", (burden(*c), t["total"]),
                    xytext=(16, -2) if c != (8, 4) else (-16, -22), textcoords="offset points", fontsize=8.5,
                    ha="left" if c != (8, 4) else "right", va="center")
    ax.set_xlim(20, 155); ax.set_ylim(0, max(T.values()) * 1.25)
    ax.set_xlabel("Weighted circuit burden $C=G_{1q}+2G_{2q}$"); ax.set_ylabel("Local Aer compile + execution time (s, 20 circuits)")
    ax.grid(True); fig.tight_layout(); fig.savefig(FIG / "Figure_8_Runtime_Burden.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Table 10 / Figure 9 (FakeNairobiV2)
if (RES / "fakenairobi_trials.csv").exists():
    fn = pd.read_csv(RES / "fakenairobi_trials.csv").sort_values("seed")
    syn = ret[(ret.dataset == "breast_cancer") & (ret.n_qubits == 4) & (ret.n_layers == 2) & np.isclose(ret.eps_2q, 1e-2)].set_index("seed").test_accuracy
    d = fn.set_index("seed").test_accuracy - syn.loc[fn.seed]
    lo, hi = boot(d.to_numpy())
    w = stats.wilcoxon(d.to_numpy(), method="approx") if np.any(d != 0) else None
    S["nairobi"] = dict(acc=summ(fn.test_accuracy), acc_min=float(fn.test_accuracy.min()), acc_max=float(fn.test_accuracy.max()),
                        train_s=summ(fn.train_seconds), eval_s=summ(fn.eval_seconds), n_trials=len(fn),
                        same_params_synthetic=summ(fn.test_accuracy_same_params_synthetic),
                        diff_vs_primary=dict(mean=float(d.mean()), lo=lo, hi=hi, p_w=float(w.pvalue) if w else 1.0),
                        layout=fn.physical_layout.iloc[0], n_test=int(fn.n_test.iloc[0]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
    ax[0].plot(fn.seed, fn.test_accuracy, "o-", color="#2a78d6", ms=6, lw=1.2, label="FakeNairobiV2 (calibrated snapshot)")
    ax[0].plot(syn.loc[fn.seed].index, syn.loc[fn.seed].values, "s--", color="#eb6834", ms=5, lw=1, label="Primary synthetic model ($\\epsilon_{2q}=10^{-2}$)")
    a = S["nairobi"]["acc"]
    ax[0].axhline(a["mean"], color="#2a78d6", lw=1, ls=":")
    ax[0].axhspan(a["mean"] - a["sd"], a["mean"] + a["sd"], color="#2a78d6", alpha=0.08)
    ax[0].set_xlabel("Split seed"); ax[0].set_ylabel("Test accuracy"); ax[0].set_ylim(0, 1.05); ax[0].grid(True)
    ax[0].set_title(f"(a) Test accuracy, mean {a['mean']:.3f} $\\pm$ {a['sd']:.3f} (SD)"); ax[0].legend(frameon=False, loc="lower left", fontsize=8.5)
    ax[1].bar(fn.seed, fn.train_seconds, 0.6, color="#2a78d6")
    ax[1].set_xlabel("Split seed"); ax[1].set_ylabel("Local training wall-clock time (s)"); ax[1].grid(axis="y")
    ax[1].set_ylim(0, fn.train_seconds.max() * 1.2)
    ax[1].set_title(f"(b) Local training time (20 SPSA iterations)")
    fig.tight_layout(); fig.savefig(FIG / "Figure_9_Calibrated_Model.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Table 11 (Pre-VDCS) and Table 12
if T:
    bc = "breast_cancer"; Aref = S["A_ref"][bc]["mean"]; T0 = min(T.values()); C0 = burden(4, 2)
    w = np.array([6, 5, 4, 2]) / 17; rows = []; S["prevdcs"] = {}
    for c in C.CONFIGS:
        A = S["vqc"][f"{bc}|{c[0]}|{c[1]}"]["mean"]
        comp = np.array([min(1, A / Aref), min(1, T0 / T[c]), fid(*c), min(1, C0 / burden(*c))])
        sc = 100 * float(w @ comp); S["prevdcs"][f"{c[0]}Q-{c[1]}L"] = dict(sA=comp[0], sT=comp[1], sN=comp[2], sC=comp[3], score=sc)
        rows.append(f"VQC ($n={c[0]}, L={c[1]}$) & {comp[0]:.4f} & {comp[1]:.4f} & {comp[2]:.4f} & {comp[3]:.4f} & \\textbf{{{sc:.2f}}} \\\\")
    S["T0"] = T0
    (OUT / "table11_prevdcs.tex").write_text("\n".join(rows) + "\n")
rows = []
for ds in C.DATASETS:
    A = S["vqc"][f"{ds}|8|4"]["mean"]; R = S["A_ref"][ds]["mean"]
    rows.append(f"{C.LABEL[ds]} & {A:.4f} & {R:.4f} & ${A - R:.4f}$ \\\\")
(OUT / "table12_breakeven.tex").write_text("\n".join(rows) + "\n")

(OUT / "summary.json").write_text(json.dumps(S, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
print(json.dumps({"A_ref": S["A_ref"], "vqc": {k: round(v["mean"], 4) for k, v in S["vqc"].items()},
                  "prevdcs": {k: round(v["score"], 2) for k, v in S.get("prevdcs", {}).items()}}, indent=1))
