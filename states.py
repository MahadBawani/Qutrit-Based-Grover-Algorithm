"""
states.py
=========
General qudit state construction utilities for the qudit Grover project.

Design goals
------------
1.  Fully general in qudit dimension `d` (d = 2 recovers ordinary qubits,
    d = 3 gives qutrits, etc.) and in the number of qudits `n`. Nothing
    here is hard-coded to d = 3, so this same module backs the d = 2
    baseline comparison and the d = 3 target case without duplication.

2.  Memory-conscious:
    - QuTiP's `Qobj` uses a sparse (CSR) backend by default, and we
      deliberately never force a dense conversion (`.full()`) internally.
    - Single-qudit basis kets |0>, |1>, ..., |d-1> are the atomic pieces
      every composite state gets built from, so they're cached with
      `functools.lru_cache`. The cache always hands back a `.copy()` of
      the stored Qobj so callers can freely mutate what they receive
      without corrupting the cached original or aliasing state between
      unrelated parts of the simulation.
    - We deliberately do NOT cache full n-qudit composite states, since
      that cache would grow as d**n and defeat the point. `uniform_superposition`
      in particular is built directly as one array rather than by tensoring
      n kets together one at a time, to avoid n-1 intermediate Qobj
      allocations.
    - `clear_state_cache()` is provided to release the basis-ket cache
      between sweeps over different dimensions `d`, which matters for the
      benchmarking work (ideal/noisy, qutrit vs qubit) where the same
      session may touch several values of d back to back.

3.  Every returned Qobj carries explicit, consistent `.dims`, matching
    QuTiP 5.x's own tensor-product convention exactly: [[d]*n, [1]] for an
    n-qudit ket (note the column side is a single [1], not [1]*n, in
    QuTiP 5.x -- this differs from older QuTiP 4.x behaviour) and
    [[d]*n, [d]*n] for an n-qudit density matrix / operator. We derive
    dims from qt.tensor()'s own output rather than hand-writing them, so
    this stays correct even if the convention shifts again -- this class
    of bug (dims mismatches breaking downstream tensor/ptrace calls) was
    a recurring issue in earlier QuTiP work on this project.

Basis-state indexing convention
--------------------------------
For n qudits you can index the computational basis two ways:
    - a length-n sequence, one index per qudit: basis_state(3, [0, 2])
    - a single flat integer in [0, d**n), decomposed into n base-d
      digits (most-significant / left-most qudit first) -- the same
      convention QuTiP itself uses when tensoring subsystems together.
      basis_state(3, 5, n=2) == basis_state(3, [1, 2], n=2)   (5 = 1*3 + 2)
"""

from __future__ import annotations

import functools
from typing import List, Optional, Sequence, Union

import numpy as np
import qutip as qt


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_dim(d: int) -> None:
    """Raise if d is not a valid qudit dimension (integer >= 2)."""
    if not isinstance(d, (int, np.integer)) or d < 2:
        raise ValueError(f"Qudit dimension d must be an integer >= 2, got {d!r}")


def _validate_index(i: int, d: int) -> None:
    """Raise if i is not a valid basis index for dimension d."""
    if not isinstance(i, (int, np.integer)) or not (0 <= i < d):
        raise ValueError(f"Basis index must satisfy 0 <= i < d={d}, got {i!r}")


def _int_to_digits(value: int, d: int, n: int) -> List[int]:
    """
    Decompose flat integer 'value' into n base-d digits, most-significant
    (left-most qudit) first. E.g. d=3, n=2, value=5 -> [1, 2].
    """
    if not (0 <= value < d ** n):
        raise ValueError(f"Flat index {value} out of range for {n} qudits of dim {d}")
    digits = []
    for _ in range(n):
        value, r = divmod(value, d)
        digits.append(r)
    return digits[::-1]


def _normalize_indices(indices: Union[int, Sequence[int]], n: int, d: int) -> List[int]:
    """
    Turn either a flat integer or a per-qudit sequence into a validated
    list of n indices, each in [0, d).
    """
    if isinstance(indices, (int, np.integer)):
        indices = [int(indices)] if n == 1 else _int_to_digits(int(indices), d, n)
    indices = list(indices)
    if len(indices) != n:
        raise ValueError(f"Expected {n} indices (one per qudit), got {len(indices)}")
    for i in indices:
        _validate_index(i, d)
    return indices


# ---------------------------------------------------------------------------
# Cached single-qudit basis kets (memory management core)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _cached_single_qudit_basis(d: int, i: int) -> qt.Qobj:
    _validate_dim(d)
    _validate_index(i, d)
    return qt.basis(d, i)


def _ket_dims(d: int, n: int) -> List[list]:
    """
    dims for an n-qudit ket, matching qt.tensor()'s own convention. Derived
    empirically rather than assumed, since this changed between QuTiP 4.x
    ([[d]*n, [1]*n]) and QuTiP 5.x ([[d]*n, [1]]) -- probing qt.tensor()
    directly means this stays correct if the convention ever shifts again.
    """
    if n == 1:
        return [[d], [1]]
    probe = qt.tensor([qt.basis(d, 0)] * n)
    return probe.dims


def clear_state_cache() -> None:
    """
    Free the single-qudit basis-ket cache.

    Call this between sweeps over different qudit dimensions (e.g.
    benchmarking d=2 vs d=3 Grover back-to-back in the same script/notebook
    session) so the cache doesn't quietly accumulate entries for every
    dimension you've ever touched.
    """
    _cached_single_qudit_basis.cache_clear()


# ---------------------------------------------------------------------------
# Basis states
# ---------------------------------------------------------------------------

def basis_state(d: int, indices: Union[int, Sequence[int]], n: int = 1) -> qt.Qobj:
    """
    Computational basis ket for n qudits of dimension d.

    Parameters
    ----------
    d : int
        Qudit dimension (2 = qubit, 3 = qutrit, ...).
    indices : int or sequence of int
        Flat integer in [0, d**n), or a length-n sequence of per-qudit
        indices. See module docstring for the indexing convention.
    n : int
        Number of qudits.

    Returns
    -------
    qt.Qobj
        Ket with dims == [[d]*n, [1]*n].
    """
    _validate_dim(d)
    idx_list = _normalize_indices(indices, n, d)

    if n == 1:
        # Copy so the caller can't mutate the cached original.
        return _cached_single_qudit_basis(d, idx_list[0]).copy()

    kets = [_cached_single_qudit_basis(d, i).copy() for i in idx_list]
    # qt.tensor already produces the correct dims for this QuTiP version;
    # no manual override needed (and manually forcing [1]*n would be
    # wrong under QuTiP 5.x's [[d]*n, [1]] convention -- see _ket_dims).
    return qt.tensor(kets)


def comp_basis(d: int) -> List[qt.Qobj]:
    """
    Full single-qudit computational basis {|0>, |1>, ..., |d-1>} as a list.
    Useful for building operators (gates, projectors) as explicit sums.
    """
    _validate_dim(d)
    return [basis_state(d, i) for i in range(d)]


# ---------------------------------------------------------------------------
# Superposition / mixed states
# ---------------------------------------------------------------------------

def uniform_superposition(d: int, n: int = 1) -> qt.Qobj:
    """
    Equal-weight superposition over all d**n computational basis states of
    n qudits of dimension d:

        |s> = (1 / sqrt(d**n)) * sum_i |i>

    This is the standard Grover initial state -- the qudit analogue of
    H^{\\otimes n} on qubits, produced instead by the generalized
    Chrestenson transform (built in gates.py) applied to |0...0>. Built
    directly as one column array rather than via repeated tensor() calls,
    which avoids d**n - 1 intermediate Qobj allocations.
    """
    _validate_dim(d)
    dim_total = d ** n
    data = np.full((dim_total, 1), 1.0 / np.sqrt(dim_total))
    return qt.Qobj(data, dims=_ket_dims(d, n))


def maximally_mixed(d: int, n: int = 1) -> qt.Qobj:
    """Maximally mixed state I / d**n for n qudits of dimension d."""
    _validate_dim(d)
    dim_total = d ** n
    rho = qt.qeye(dim_total) / dim_total
    rho.dims = [[d] * n, [d] * n]
    return rho


def density_from_ket(ket: qt.Qobj) -> qt.Qobj:
    """Pure-state density matrix rho = |psi><psi|, preserving dims."""
    if not ket.isket:
        raise ValueError("density_from_ket expects a ket Qobj (type='ket')")
    rho = ket * ket.dag()
    rho.dims = [ket.dims[0], ket.dims[0]]
    return rho


def random_pure_state(d: int, n: int = 1, seed: Optional[int] = None) -> qt.Qobj:
    """Haar-random pure state of n qudits of dimension d."""
    _validate_dim(d)
    dim_total = d ** n
    state = qt.rand_ket(dim_total, seed=seed)
    state.dims = _ket_dims(d, n)
    return state


def random_density_matrix(
    d: int, n: int = 1, seed: Optional[int] = None, rank: Optional[int] = None
) -> qt.Qobj:
    """
    Random density matrix of n qudits of dimension d (via qutip.rand_dm).
    `rank` restricts to a lower-rank / more-mixed random state; None means
    full rank.
    """
    _validate_dim(d)
    dim_total = d ** n
    rho = qt.rand_dm(dim_total, seed=seed, rank=rank)
    rho.dims = [[d] * n, [d] * n]
    return rho


# ---------------------------------------------------------------------------
# Generalized entangled states (qudit GHZ / Bell)
# ---------------------------------------------------------------------------

def ghz_state(d: int, n: int = 2) -> qt.Qobj:
    """
    Generalized GHZ state for n qudits of dimension d:

        (1 / sqrt(d)) * sum_{k=0}^{d-1} |k>|k>...|k>   (n copies)

    n=2, d=2 reduces to the ordinary Bell state |Phi+>.
    """
    _validate_dim(d)
    if n < 2:
        raise ValueError("GHZ state requires n >= 2 qudits")
    total = qt.Qobj(np.zeros((d ** n, 1)), dims=_ket_dims(d, n))
    for k in range(d):
        flat_index = sum(k * d ** (n - 1 - j) for j in range(n))
        total = total + basis_state(d, flat_index, n=n)
    return total.unit()


def qudit_bell_state(d: int, p: int = 0, q: int = 0) -> qt.Qobj:
    """
    Generalized two-qudit Bell state from the Weyl-Heisenberg
    (clock-and-shift) construction, labeled by (p, q) in Z_d x Z_d:

        |Phi_pq> = (1/sqrt(d)) * sum_k exp(2*pi*i*p*k/d) |k> tensor |(k+q) mod d>

    (p, q) = (0, 0) gives the standard maximally-entangled state; varying
    (p, q) over Z_d x Z_d gives the full orthonormal Bell basis for two
    qudits -- reusable later for qudit teleportation / entanglement-swapping
    checks in the same spirit as the earlier qubit work on this project.
    """
    _validate_dim(d)
    _validate_index(p, d)
    _validate_index(q, d)
    total = qt.Qobj(np.zeros((d * d, 1), dtype=complex), dims=_ket_dims(d, 2))
    for k in range(d):
        phase = np.exp(2j * np.pi * p * k / d)
        flat_index = k * d + ((k + q) % d)
        total = total + phase * basis_state(d, flat_index, n=2)
    return total.unit()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def purity(rho: qt.Qobj) -> float:
    """Tr(rho^2). Accepts either a ket (auto-promoted) or a density matrix."""
    if rho.isket:
        rho = density_from_ket(rho)
    return float((rho * rho).tr().real)


def is_valid_state(state: qt.Qobj, atol: float = 1e-9) -> bool:
    """
    Sanity-check a ket or density matrix: correct normalization, and for
    density matrices, Hermiticity + positive semi-definiteness + unit
    trace. A cheap guardrail against dims/normalization bugs, which were a
    recurring source of errors in earlier QuTiP work on this project.
    """
    if state.isket:
        return abs(state.norm() - 1.0) < atol
    if state.isoper:
        herm_ok = (state - state.dag()).norm() < atol
        trace_ok = abs(state.tr() - 1.0) < atol
        eigs = state.eigenenergies()
        pos_ok = bool(np.all(eigs > -atol))
        return bool(herm_ok and trace_ok and pos_ok)
    return False
