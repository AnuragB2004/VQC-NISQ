"""Qiskit Aer experiments on the final circuit architecture.

mode 'timing'   (Table 9, Figure 8): for each configuration, 20 circuits (the trained
    seed-42 Breast Cancer parameters bound to the first 20 test samples) are transpiled
    (basis {rz, sx, x, cx}, linear coupling map, optimization_level=2, SABRE layout and
    routing, seed_transpiler=42) and executed on AerSimulator with depolarizing noise
    (eps_1q = 1e-3 on rz/sx/x, eps_2q = 1e-2 on cx) and 1024 shots. Compile and execution
    wall-clock times are measured separately; 1 warm-up + 10 timed repetitions.
mode 'nairobi'  (Table 10, Figure 9): Breast Cancer, n = 4, L = 2, the same 15 splits (seeds 42-56). Algorithm 1
    is run with every loss evaluation estimated from 1024-shot counts on
    AerSimulator.from_backend(FakeNairobiV2()) (calibrated gate, readout and thermal-relaxation
    noise of the offline snapshot); the test partition is evaluated the same way.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import common as C
import vqc_core as V

MODE = sys.argv[1]
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results"); OUT.mkdir(parents=True, exist_ok=True)


def param_circuit(n, L):
    x = ParameterVector("x", n)
    th = ParameterVector("t", V.n_params(n, L))
    qc = QuantumCircuit(n, 1)
    for q in range(n): qc.ry(np.pi * x[q], q)
    for q in range(n): qc.ry(th[q], q)
    k = n
    for r in range(L):
        for q in range(n): qc.ry(th[k + q], q)
        for q in range(n): qc.rz(th[k + n + q], q)
        k += 2 * n
        for q in range(n - 1): qc.cx(q, q + 1)
    qc.measure(0, 0)
    return qc, x, th


def z_from_counts(counts, S):
    n0 = counts.get("0", 0)
    return (2 * n0 - S) / S


if MODE == "timing":
    trials = pd.read_csv(OUT / "vqc_trials.csv")
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(C.EPS_1Q, 1), ["rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(C.EPS_2Q_PRIMARY, 2), ["cx"])
    rows, meta = [], {}
    for n, L in C.CONFIGS:
        Ftr, Fte, ytr, yte = C.matched_data("breast_cancer", 42, n)
        theta, _, _ = V.spsa_train(Ftr, ytr, n, L, C.EPS_1Q, C.EPS_2Q_PRIMARY, 42, n_iter=C.N_ITER, S=C.SHOTS)
        qc, xp, tp = param_circuit(n, L)
        circs = [qc.assign_parameters({**dict(zip(xp, Fte[i])), **dict(zip(tp, theta))}) for i in range(20)]
        cmap = CouplingMap.from_line(n)
        sim = AerSimulator(noise_model=nm)
        kw = dict(basis_gates=["rz", "sx", "x", "cx"], coupling_map=cmap, optimization_level=2,
                  layout_method="sabre", routing_method="sabre", seed_transpiler=42)
        tq = transpile(circs, **kw); sim.run(tq, shots=C.SHOTS, seed_simulator=0).result()  # warm-up
        for rep in range(10):
            t0 = time.perf_counter(); tq = transpile(circs, **kw); tc = time.perf_counter() - t0
            t0 = time.perf_counter(); res = sim.run(tq, shots=C.SHOTS, seed_simulator=rep).result(); te = time.perf_counter() - t0
            rows.append(dict(n_qubits=n, n_layers=L, repetition=rep, compile_s=tc, execute_s=te))
        ops = [dict(c.count_ops()) for c in tq]
        meta[f"{n}Q-{L}L"] = dict(method=res.results[0].metadata.get("method"),
                                  transpiled_depth_mean=float(np.mean([c.depth() for c in tq])),
                                  transpiled_cx_mean=float(np.mean([o.get("cx", 0) for o in ops])),
                                  transpiled_1q_mean=float(np.mean([sum(v for k, v in o.items() if k in ("rz", "sx", "x")) for o in ops])),
                                  logical_counts=dict(qc.count_ops()))
        d = pd.DataFrame(rows[-10:])
        print(f"{n}Q-{L}L compile {d.compile_s.mean():.4f}±{d.compile_s.std():.4f}  exec {d.execute_s.mean():.4f}±{d.execute_s.std():.4f}  {meta[f'{n}Q-{L}L']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "aer_timing.csv", index=False)
    (OUT / "aer_timing_meta.json").write_text(json.dumps(meta, indent=2))

elif MODE == "nairobi":
    from qiskit.providers.fake_provider import FakeNairobiV2
    backend = FakeNairobiV2()
    sim = AerSimulator.from_backend(backend)
    n, L, S = 4, 2, C.SHOTS
    qc, xp, tp = param_circuit(n, L)
    tqc = transpile(qc, backend=backend, optimization_level=2, layout_method="sabre",
                    routing_method="sabre", seed_transpiler=42)
    layout = tqc.layout.final_index_layout() if tqc.layout is not None else None
    rows = []
    for seed in C.SEEDS:
        Ftr, Fte, ytr, yte = C.matched_data("breast_cancer", seed, n)
        rng = np.random.default_rng(seed)
        P = V.n_params(n, L)
        theta = rng.uniform(-np.pi, np.pi, P)
        sim_seed = [int(seed) * 1000]

        def zhat(th, X):
            binds = {p: list(X[:, i]) for i, p in enumerate(xp)}
            binds.update({p: [float(th[i])] * len(X) for i, p in enumerate(tp)})
            sim_seed[0] += 1
            res = sim.run(tqc, shots=S, parameter_binds=[binds], seed_simulator=sim_seed[0]).result()
            return np.array([z_from_counts(res.get_counts(i), S) for i in range(len(X))])

        t0 = time.perf_counter()
        for k in range(C.N_ITER):
            ak = 0.05 / (10 + k + 1) ** 0.602; ck = 0.10 / (k + 1) ** 0.101
            delta = rng.choice([-1.0, 1.0], size=P)
            lp = np.mean((zhat(theta + ck * delta, Ftr) - ytr) ** 2)
            lm = np.mean((zhat(theta - ck * delta, Ftr) - ytr) ** 2)
            theta = theta - ak * (lp - lm) / (2 * ck) * delta
        t_train = time.perf_counter() - t0
        t0 = time.perf_counter(); z = zhat(theta, Fte); t_eval = time.perf_counter() - t0
        acc = float(np.mean(V.predict_labels(z) == yte))
        # same trained parameters under the synthetic Pauli-propagation model, for reference
        acc_syn, _ = V.evaluate(theta, Fte, yte, n, L, C.EPS_1Q, C.EPS_2Q_PRIMARY, np.random.default_rng([seed, 9]), S)
        rows.append(dict(seed=seed, test_accuracy=acc, test_accuracy_same_params_synthetic=acc_syn,
                         n_train=len(ytr), n_test=len(yte), train_seconds=t_train, eval_seconds=t_eval,
                         physical_layout=str(layout), backend=backend.name))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "fakenairobi_trials.csv", index=False)
