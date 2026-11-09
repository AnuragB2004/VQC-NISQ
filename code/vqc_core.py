"""Core of the primary VQC simulation (Algorithm 1 of the manuscript).

Circuit (Schrodinger order), n qubits, L layers:
    R_Y(pi x_q)                         q = 0..n-1   data encoding (not trainable)
    R_Y(theta_init_q)                   q = 0..n-1   initial trainable layer
    for r = 1..L:
        R_Y(theta_Y_{q,r}) R_Z(theta_Z_{q,r})   q = 0..n-1
        CNOT(q, q+1)                            q = 0..n-2   open nearest-neighbour chain
    readout: <Z_0>

Noise: every one-qubit gate is followed by a one-qubit depolarizing channel with
parameter eps_1q and every CNOT by a two-qubit depolarizing channel with parameter
eps_2q, using the Qiskit convention E(rho) = (1-p) rho + p Tr(rho) I/d. Under this
convention every non-identity Pauli string on the channel's support is scaled by (1-p).

Expectation values are computed exactly in the Heisenberg picture: the observable Z_0
is expanded in the n-qubit Pauli basis (a (4,)*n coefficient tensor) and propagated
backwards through the trainable part of the circuit with the Pauli transfer matrices of
each gate and the adjoint (= self) depolarizing channels. The data-dependent encoding
layer acting on |0> produces a product state with Bloch vector
(1-eps_1q)(sin(pi x), 0, cos(pi x)) on each qubit, so the expectation of the propagated
observable is a product contraction that is vectorised over samples.

Finite-shot readout: N_0 ~ Binomial(S, (1+m)/2) and z_hat = (2 N_0 - S)/S (Algorithm 1).
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ Pauli algebra
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [_I, _X, _Y, _Z]  # index order I, X, Y, Z


def _heis_ptm_1q(U):
    """M[P, Q] with U^dag P U = sum_Q M[P,Q] Q."""
    M = np.empty((4, 4))
    for a, P in enumerate(PAULIS):
        O = U.conj().T @ P @ U
        for b, Q in enumerate(PAULIS):
            M[a, b] = np.real(np.trace(Q @ O)) / 2
    return M


def _heis_ptm_2q(U):
    M = np.empty((4, 4, 4, 4))
    for a, Pa in enumerate(PAULIS):
        for b, Pb in enumerate(PAULIS):
            O = U.conj().T @ np.kron(Pa, Pb) @ U
            for c, Qa in enumerate(PAULIS):
                for d, Qb in enumerate(PAULIS):
                    M[a, b, c, d] = np.real(np.trace(np.kron(Qa, Qb) @ O)) / 4
    return M


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)


# qubit q is the control (first tensor factor), q+1 the target
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
M_CNOT = _heis_ptm_2q(CNOT)

# Closed forms (checked against _heis_ptm_1q in the tests)
def m_ry(t):
    c, s = np.cos(t), np.sin(t)
    # U^dag X U = c X + s Z ; U^dag Z U = -s X + c Z ; Y invariant
    return np.array([[1, 0, 0, 0], [0, c, 0, s], [0, 0, 1, 0], [0, -s, 0, c]])


def m_rz(t):
    c, s = np.cos(t), np.sin(t)
    # U^dag X U = c X - s Y ; U^dag Y U = s X + c Y ; Z invariant
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def n_params(n, L):
    return n * (2 * L + 1)


def unpack(theta, n, L):
    theta = np.asarray(theta, float)
    init = theta[:n]
    rest = theta[n:].reshape(L, 2, n)  # [layer, (Y,Z), qubit]
    return init, rest[:, 0, :], rest[:, 1, :]


def gate_list(theta, n, L):
    """Trainable gates in Schrodinger order: ('1q', q, M) or ('cx', q, None)."""
    init, ty, tz = unpack(theta, n, L)
    g = [("1q", q, m_ry(init[q])) for q in range(n)]
    for r in range(L):
        for q in range(n):
            g.append(("1q", q, m_ry(ty[r, q])))
            g.append(("1q", q, m_rz(tz[r, q])))
        for q in range(n - 1):
            g.append(("cx", q, None))
    return g




def _dep_mask_1q(n, q, lam):
    s = np.ones(4)
    s[1:] = lam
    shape = [1] * n
    shape[q] = 4
    return s.reshape(shape)


def _dep_mask_2q(n, q, lam):
    s = np.full((4, 4), lam)
    s[0, 0] = 1.0
    shape = [1] * n
    shape[q] = 4
    shape[q + 1] = 4
    return s.reshape(shape)


def propagate(theta, n, L, eps1, eps2):
    """Heisenberg-propagate Z_0 backwards through the noisy trainable circuit."""
    c = np.zeros((4,) * n)
    idx = [0] * n
    idx[0] = 3
    c[tuple(idx)] = 1.0
    lam1, lam2 = 1.0 - eps1, 1.0 - eps2
    masks1 = [_dep_mask_1q(n, q, lam1) for q in range(n)]
    masks2 = [_dep_mask_2q(n, q, lam2) for q in range(n - 1)]
    for kind, q, M in reversed(gate_list(theta, n, L)):
        if kind == "1q":
            c = c * masks1[q]                       # adjoint noise channel
            c = np.moveaxis(np.tensordot(c, M, axes=([q], [0])), -1, q)  # U^dag . U
        else:
            c = c * masks2[q]
            c = np.tensordot(c, M_CNOT, axes=([q, q + 1], [0, 1]))
            c = np.moveaxis(c, [-2, -1], [q, q + 1])
    return c


def expectation_from_observable(c, X, eps1):
    """<O> for each row of X on the noisy encoded product state."""
    lam = 1.0 - eps1
    N, n = X.shape
    a = np.pi * X
    V = np.stack([np.ones_like(a), lam * np.sin(a), np.zeros_like(a), lam * np.cos(a)], axis=-1)  # N,n,4
    # contract qubit 0 first: result (N, 4, ..., 4) with n-1 trailing axes
    R = np.tensordot(V[:, 0, :], c, axes=([1], [0]))
    for q in range(1, n):
        R = np.einsum("nj...,nj->n...", R, V[:, q, :])
    return R


def expectations(theta, X, n, L, eps1, eps2):
    return expectation_from_observable(propagate(theta, n, L, eps1, eps2), X, eps1)


def shots(m, S, rng):
    p = np.clip((1.0 + m) / 2.0, 0.0, 1.0)
    N0 = rng.binomial(S, p)
    return (2.0 * N0 - S) / S


def predict_labels(zhat):
    return np.where(zhat >= 0, 1, -1)


def spsa_train(X, y, n, L, eps1, eps2, seed, n_iter=20, S=1024,
               a=0.05, c=0.10, A=10, alpha=0.602, gamma=0.101, history=False):
    """Algorithm 1. Returns theta*, rng (to continue the stream), and optional loss history."""
    rng = np.random.default_rng(seed)
    P = n_params(n, L)
    theta = rng.uniform(-np.pi, np.pi, P)
    hist = []
    for k in range(n_iter):
        ak = a / (A + k + 1) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=P)
        losses = {}
        for s in (+1, -1):
            th = theta + s * ck * delta
            m = expectations(th, X, n, L, eps1, eps2)
            z = shots(m, S, rng)
            losses[s] = float(np.mean((z - y) ** 2))
        g = (losses[+1] - losses[-1]) / (2 * ck) * delta  # Delta^{-1} = Delta for +-1 entries
        theta = theta - ak * g
        if history:
            hist.append(0.5 * (losses[+1] + losses[-1]))
    return theta, rng, hist


def evaluate(theta, X, y, n, L, eps1, eps2, rng, S=1024):
    m = expectations(theta, X, n, L, eps1, eps2)
    z = shots(m, S, rng)
    return float(np.mean(predict_labels(z) == y)), m
