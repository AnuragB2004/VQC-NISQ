#!/usr/bin/env python3
"""Regenerate the analytical figures (Figures 4, 5 and 8) from the revised circuit model.

Circuit model (Eqs. 2-5 of the manuscript, logical pre-transpilation counts):
    G_1q(n, L) = n (2L + 2)      n encoding R_Y + n initial trainable R_Y + 2nL layer R_Y/R_Z
    G_2q(n, L) = (n - 1) L       open nearest-neighbour CNOT chain per layer
    P(n, L)    = n (2L + 1)      trainable parameters
    D(n, L)   <= 2 + L (n + 1)   logical-depth upper bound
    C(n, L)    = G_1q + 2 G_2q   weighted circuit burden
    F          = (1 - eps_1q)^G_1q (1 - eps_2q)^G_2q   gate-survival proxy

Local timing values (Figure 8) are copied from Table 9 of the manuscript; they are
NOT recomputed here.

Usage:  python3 make_analytical_figures.py [output_dir]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)

EPS_1Q = 1e-3
EPS_2Q_GRID = [1e-3, 1e-2, 1e-1]
CONFIGS = [(4, 2), (8, 2), (8, 4)]
LABELS = {c: f"{c[0]}Q–{c[1]}L" for c in CONFIGS}
COLORS = {(4, 2): "#2a78d6", (8, 2): "#eb6834", (8, 4): "#1baf7a"}
MARKERS = {(4, 2): "o", (8, 2): "s", (8, 4): "D"}

# Table 9 (local Aer wall-clock, seconds per 20 circuits)
TIMING = {(4, 2): (0.030, 0.020), (8, 2): (0.028, 1.194), (8, 4): (0.034, 1.630)}


def g1q(n, L):
    return n * (2 * L + 2)


def g2q(n, L):
    return (n - 1) * L


def params(n, L):
    return n * (2 * L + 1)


def depth_bound(n, L):
    return 2 + L * (n + 1)


def burden(n, L):
    return g1q(n, L) + 2 * g2q(n, L)


def fidelity(n, L, e2, e1=EPS_1Q):
    return (1 - e1) ** g1q(n, L) * (1 - e2) ** g2q(n, L)


plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.labelcolor": "#222222",
    "xtick.color": "#444444", "ytick.color": "#444444",
    "grid.color": "#dddddd", "grid.linewidth": 0.6,
})

# ---------------------------------------------------------------- self-check
EXPECTED = {(4, 2): (24, 6, 20, 12, 36), (8, 2): (48, 14, 40, 20, 76), (8, 4): (80, 28, 72, 38, 136)}
for c, exp in EXPECTED.items():
    got = (g1q(*c), g2q(*c), params(*c), depth_bound(*c), burden(*c))
    assert got == exp, (c, got, exp)
assert [round(100 * fidelity(*c, 1e-2), 2) for c in CONFIGS] == [91.91, 82.80, 69.67]

print("config   G1q G2q  P  D<=  C    F(1e-3)  F(1e-2)  F(1e-1)")
for c in CONFIGS:
    f = [100 * fidelity(*c, e) for e in EPS_2Q_GRID]
    print(f"{LABELS[c]:7s} {g1q(*c):4d} {g2q(*c):3d} {params(*c):3d} {depth_bound(*c):3d} {burden(*c):4d}"
          f"  {f[0]:7.2f}  {f[1]:7.2f}  {f[2]:7.2f}")

n_grid = np.arange(2, 13)
L_grid = np.arange(1, 7)
NN, LL = np.meshgrid(n_grid, L_grid)

# ---------------------------------------------------------------- Figure 4
fig = plt.figure(figsize=(12, 4.8))
ax = fig.add_subplot(1, 2, 1, projection="3d")
CC = burden(NN, LL)
ax.plot_surface(NN, LL, CC, cmap="Blues", alpha=0.55, linewidth=0.3, edgecolor="#9ec5f4",
                antialiased=True, vmin=0, vmax=CC.max() * 1.1)
for c in CONFIGS:
    ax.scatter(*c, burden(*c), s=60, color=COLORS[c], marker=MARKERS[c],
               edgecolor="white", linewidth=1.2, depthshade=False, zorder=10)
    ax.text(c[0], c[1], burden(*c) + 18, f"{LABELS[c]}\nC={burden(*c)}", fontsize=8.5,
            ha="center", color="#222222", zorder=11)
ax.set_xlabel("Qubits, $n$")
ax.set_ylabel("Layers, $L$")
ax.set_zlabel("$C=G_{1q}+2G_{2q}$")
ax.set_xticks([2, 4, 6, 8, 10, 12])
ax.set_yticks(range(1, 7))
ax.view_init(elev=24, azim=-58)
ax.set_title("(a) Weighted circuit-burden surface", pad=4)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))

ax2 = fig.add_subplot(1, 2, 2)
for c in CONFIGS:
    d = depth_bound(*c)
    ax2.scatter(g1q(*c), g2q(*c), s=18 * d, color=COLORS[c], marker=MARKERS[c],
                edgecolor="white", linewidth=1.5, zorder=3)
    ax2.annotate(f"{LABELS[c]}\n$G_{{1q}}$={g1q(*c)}, $G_{{2q}}$={g2q(*c)}\n$D\\leq{d}$",
                 (g1q(*c), g2q(*c)), xytext=(14, -4), textcoords="offset points",
                 fontsize=8.5, va="center", color="#222222")
ax2.set_xlim(10, 110)
ax2.set_ylim(0, 34)
ax2.set_xlabel("One-qubit gate count $G_{1q}=n(2L+2)$")
ax2.set_ylabel("Two-qubit gate count $G_{2q}=(n-1)L$")
ax2.grid(True)
ax2.set_title("(b) Evaluated configurations (marker area $\\propto$ depth bound)")
fig.tight_layout()
fig.savefig(OUT / "Figure_4_Complexity_3D.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 5
fig = plt.figure(figsize=(12, 4.8))
ax = fig.add_subplot(1, 2, 1, projection="3d")
FF = 100 * fidelity(NN, LL, 1e-2)
ax.plot_surface(NN, LL, FF, cmap="Blues_r", alpha=0.55, linewidth=0.3, edgecolor="#9ec5f4",
                vmin=FF.min() - 15, vmax=100)
for c in CONFIGS:
    f = 100 * fidelity(*c, 1e-2)
    ax.scatter(*c, f, s=60, color=COLORS[c], marker=MARKERS[c], edgecolor="white",
               linewidth=1.2, depthshade=False, zorder=10)
    ax.text(c[0] + {(4, 2): 0, (8, 2): -1.8, (8, 4): 1.8}[c], c[1], f + 6, f"{LABELS[c]}\n{f:.2f}%", fontsize=8.5, ha="center",
            color="#222222", zorder=11)
ax.set_xlabel("Qubits, $n$")
ax.set_ylabel("Layers, $L$")
ax.set_zlabel("Gate-survival proxy $F$ (%)")
ax.set_xticks([2, 4, 6, 8, 10, 12])
ax.set_yticks(range(1, 7))
ax.view_init(elev=24, azim=-58)
ax.set_title("(a) $F(n,L)$ at $\\epsilon_{1q}=10^{-3}$, $\\epsilon_{2q}=10^{-2}$", pad=4)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))

ax2 = fig.add_subplot(1, 2, 2)
for c in CONFIGS:
    f = [100 * fidelity(*c, e) for e in EPS_2Q_GRID]
    ax2.plot(EPS_2Q_GRID, f, color=COLORS[c], marker=MARKERS[c], markersize=7, linewidth=2,
             markeredgecolor="white", markeredgewidth=1.2, label=LABELS[c])
    ax2.annotate(f"{f[-1]:.1f}%", (EPS_2Q_GRID[-1], f[-1]), xytext=(7, 0),
                 textcoords="offset points", va="center", fontsize=8.5, color="#222222")
    ax2.annotate(f"{f[1]:.2f}%", (EPS_2Q_GRID[1], f[1]), xytext=(-8, -12 if c != (4, 2) else 8),
                 textcoords="offset points", ha="right", va="center", fontsize=8.5, color="#222222")
ax2.set_xscale("log")
ax2.set_xlim(7e-4, 1.8e-1)
ax2.set_ylim(0, 100)
ax2.set_xlabel("Two-qubit depolarizing error rate $\\epsilon_{2q}$ ($\\epsilon_{1q}=10^{-3}$)")
ax2.set_ylabel("Gate-survival proxy $F$ (%)")
ax2.grid(True, which="major")
ax2.legend(frameon=False, loc="lower left")
ax2.set_title("(b) Error-rate slices for evaluated configurations")
fig.tight_layout()
fig.savefig(OUT / "Figure_5_Noise_3D.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 8
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for c in CONFIGS:
    tc, te = TIMING[c]
    tot = tc + te
    share = 100 * te / tot
    ax.scatter(burden(*c), tot, s=4 * burden(*c), color=COLORS[c], marker=MARKERS[c],
               edgecolor="white", linewidth=1.5, zorder=3)
    off = (16, -2) if c != (8, 4) else (-16, -26)
    ax.annotate(f"{LABELS[c]} (C={burden(*c)})\n{tot:.3f} s, {share:.1f}% execution", (burden(*c), tot),
                xytext=off, textcoords="offset points", fontsize=8.5,
                ha="left" if c != (8, 4) else "right", va="center", color="#222222")
ax.set_xlim(20, 155)
ax.set_ylim(-0.1, 2.0)
ax.set_xlabel("Weighted circuit burden $C=G_{1q}+2G_{2q}$")
ax.set_ylabel("Local compile + execution time (s, 20 circuits)")
ax.grid(True)
fig.tight_layout()
fig.savefig(OUT / "Figure_8_Runtime_Burden.png", bbox_inches="tight")
plt.close(fig)
print("written:", *(p.name for p in sorted(OUT.glob("Figure_[458]_*.png"))))
