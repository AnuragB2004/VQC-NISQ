"""Validate the Pauli-propagation simulator against Qiskit Aer density-matrix simulation."""
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import vqc_core as V


def qiskit_circuit(x, theta, n, L):
    init, ty, tz = V.unpack(theta, n, L)
    qc = QuantumCircuit(n)
    for q in range(n): qc.ry(np.pi * x[q], q)
    for q in range(n): qc.ry(init[q], q)
    for r in range(L):
        for q in range(n):
            qc.ry(ty[r, q], q); qc.rz(tz[r, q], q)
        for q in range(n - 1): qc.cx(q, q + 1)
    return qc


def aer_expectation(x, theta, n, L, e1, e2):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(e1, 1), ["ry", "rz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ["cx"])
    qc = qiskit_circuit(x, theta, n, L)
    qc.save_density_matrix()
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    rho = sim.run(qc).result().data()["density_matrix"]
    return float(np.real(rho.expectation_value(SparsePauliOp("I" * (n - 1) + "Z"))))


# closed-form PTMs equal the numerically derived ones
for t in np.linspace(-3, 3, 7):
    assert np.allclose(V.m_ry(t), V._heis_ptm_1q(V.ry(t)))
    assert np.allclose(V.m_rz(t), V._heis_ptm_1q(V.rz(t)))
print("PTM closed forms OK")

rng = np.random.default_rng(0)
worst = 0
for n, L in [(4, 2), (8, 2), (8, 4)]:
    for e1, e2 in [(0, 0), (1e-3, 1e-2), (1e-3, 1e-1), (5e-2, 2e-1)]:
        th = rng.uniform(-np.pi, np.pi, V.n_params(n, L))
        X = rng.uniform(0, 1, (5, n))
        mine = V.expectations(th, X, n, L, e1, e2)
        ref = np.array([aer_expectation(x, th, n, L, e1, e2) for x in X])
        err = np.max(np.abs(mine - ref)); worst = max(worst, err)
        print(f"n={n} L={L} eps=({e1},{e2})  max|diff|={err:.2e}   e.g. {mine[0]:+.6f} vs {ref[0]:+.6f}")
assert worst < 1e-9, worst
t = time.perf_counter(); V.expectations(th, rng.uniform(0, 1, (455, 8)), 8, 4, 1e-3, 1e-2)
print(f"ALL MATCH (worst {worst:.1e}); 8Q-4L, 455 samples: {time.perf_counter()-t:.3f} s")
