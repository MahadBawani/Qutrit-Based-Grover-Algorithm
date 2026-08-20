"""
gates.py
========
General qudit gate library for the qudit Grover project. Every gate is
parameterized by dimension `d` and "collapses" to the familiar qubit gate
at d=2 (shift_gate(2) == X, clock_gate(2) == Z, chrestenson_gate(2) == H,
csum_gate(2) == CNOT, cz_gate(2) == CZ) -- there's no d=3-specific code
anywhere, so the same functions back the qutrit target case and the qubit
baseline comparison.

Conventions
-----------
- omega = exp(2*pi*i/d), the primitive d-th root of unity.
- All gates act on states built by states.py, and dims are derived by
  probing qt.tensor()/qt.qeye() directly (never hand-written), for the
  same reason documented in states.py: QuTiP's ket/operator dims format
  can differ across versions, and hardcoding it is how that class of bug
  gets reintroduced.
- `embed()` places a 1- or 2-qudit gate at a given position (or pair of
  positions) inside an m-qudit register, tensoring identities everywhere
  else. This is what makes the library "register-general": you write a
  gate once for its local Hilbert space and place it anywhere in an
  m-qudit circuit without rewriting it.

Memory management
------------------
Base single- and two-qudit gate matrices (shift, clock, Chrestenson, CSUM,
CZ, SWAP) are cached per-dimension with `functools.lru_cache`, exactly
like the basis kets in states.py -- these are small, reused constantly
while assembling oracles/diffusion operators, and are cheap to keep
around. As with states.py, the cache always hands back a `.copy()`, so a
caller who scales, exponentiates, or otherwise mutates a returned gate in
place can never corrupt what the next caller receives. `embed()` results
(full m-qudit operators) are NOT cached -- those scale as d**m and are
usually only needed once per circuit position, so caching them would
trade a small compute saving for a memory cost that grows with system
size, which is the wrong trade for this project's register sizes.
Use `clear_gate_cache()` alongside `states.clear_state_cache()` when
sweeping over multiple dimensions `d` in the same session.
"""

from __future__ import annotations

import functools
from typing import List, Sequence, Union

import numpy as np
import qutip as qt


# ---------------------------------------------------------------------------
# Validation (mirrors states.py)
# ---------------------------------------------------------------------------

def _validate_dim(d: int) -> None:
    if not isinstance(d, (int, np.integer)) or d < 2:
        raise ValueError(f"Qudit dimension d must be an integer >= 2, got {d!r}")


def _validate_position(pos: int, m: int) -> None:
    if not isinstance(pos, (int, np.integer)) or not (0 <= pos < m):
        raise ValueError(f"Position must satisfy 0 <= pos < m={m}, got {pos!r}")


def _op_dims(d: int, n: int) -> List[list]:
    """
    dims for an n-qudit operator, probed directly from qt.tensor() rather
    than hand-written -- same rationale as states._ket_dims().
    """
    if n == 1:
        return [[d], [d]]
    probe = qt.tensor([qt.qeye(d)] * n)
    return probe.dims


def omega(d: int) -> complex:
    """Primitive d-th root of unity, exp(2*pi*i/d)."""
    _validate_dim(d)
    return np.exp(2j * np.pi / d)


# ---------------------------------------------------------------------------
# Cached base gates (single- and two-qudit, memory management core)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _cached_shift_gate(d: int) -> qt.Qobj:
    """Generalized Pauli-X / cyclic shift: X_d|k> = |(k+1) mod d>."""
    data = np.zeros((d, d), dtype=complex)
    for k in range(d):
        data[(k + 1) % d, k] = 1.0
    return qt.Qobj(data, dims=[[d], [d]])


@functools.lru_cache(maxsize=None)
def _cached_clock_gate(d: int) -> qt.Qobj:
    """Generalized Pauli-Z / clock gate: Z_d|k> = omega^k |k>."""
    w = omega(d)
    data = np.diag([w ** k for k in range(d)]).astype(complex)
    return qt.Qobj(data, dims=[[d], [d]])


@functools.lru_cache(maxsize=None)
def _cached_chrestenson_gate(d: int) -> qt.Qobj:
    """
    Generalized Hadamard / discrete Fourier transform:
        (H_d)_{jk} = (1/sqrt(d)) * omega^{jk}
    Collapses exactly to the ordinary Hadamard at d=2. This is the gate
    that turns |0> into the uniform superposition used to seed Grover
    (states.uniform_superposition), applied register-wide via embed().
    """
    w = omega(d)
    j, k = np.meshgrid(np.arange(d), np.arange(d), indexing="ij")
    data = (w ** (j * k)) / np.sqrt(d)
    return qt.Qobj(data.astype(complex), dims=[[d], [d]])


@functools.lru_cache(maxsize=None)
def _cached_phase_gate(d: int, k: int, m: int) -> qt.Qobj:
    """
    Single basis-state phase gate: multiplies |k> by omega^m, identity
    elsewhere. Building block for oracle phase kicks.
    """
    w = omega(d)
    data = np.eye(d, dtype=complex)
    data[k, k] = w ** m
    return qt.Qobj(data, dims=[[d], [d]])


@functools.lru_cache(maxsize=None)
def _cached_csum_gate(d: int) -> qt.Qobj:
    """
    Generalized CNOT / controlled-sum on two qudits:
        CSUM |a>|b> = |a>|(a+b) mod d>
    Collapses exactly to CNOT at d=2.
    """
    dim2 = d * d
    data = np.zeros((dim2, dim2), dtype=complex)
    for a in range(d):
        for b in range(d):
            in_idx = a * d + b
            out_idx = a * d + ((a + b) % d)
            data[out_idx, in_idx] = 1.0
    return qt.Qobj(data, dims=_op_dims(d, 2))


@functools.lru_cache(maxsize=None)
def _cached_cz_gate(d: int) -> qt.Qobj:
    """
    Generalized controlled-Z on two qudits (diagonal, entangling phase):
        CZ_d |a>|b> = omega^{a*b} |a>|b>
    Collapses exactly to CZ at d=2.
    """
    w = omega(d)
    dim2 = d * d
    diag_entries = [w ** (a * b) for a in range(d) for b in range(d)]
    data = np.diag(diag_entries).astype(complex)
    return qt.Qobj(data, dims=_op_dims(d, 2))


@functools.lru_cache(maxsize=None)
def _cached_swap_gate(d: int) -> qt.Qobj:
    """Generalized SWAP on two qudits: SWAP|a>|b> = |b>|a>."""
    dim2 = d * d
    data = np.zeros((dim2, dim2), dtype=complex)
    for a in range(d):
        for b in range(d):
            in_idx = a * d + b
            out_idx = b * d + a
            data[out_idx, in_idx] = 1.0
    return qt.Qobj(data, dims=_op_dims(d, 2))


def clear_gate_cache() -> None:
    """
    Free all cached base gate matrices. Call alongside
    states.clear_state_cache() between sweeps over different qudit
    dimensions to keep memory flat during long benchmark runs.
    """
    _cached_shift_gate.cache_clear()
    _cached_clock_gate.cache_clear()
    _cached_chrestenson_gate.cache_clear()
    _cached_phase_gate.cache_clear()
    _cached_csum_gate.cache_clear()
    _cached_cz_gate.cache_clear()
    _cached_swap_gate.cache_clear()


# ---------------------------------------------------------------------------
# Public single-qudit gates (each returns a fresh copy of the cached gate)
# ---------------------------------------------------------------------------

def shift_gate(d: int) -> qt.Qobj:
    """Generalized X_d: X_d|k> = |(k+1) mod d>."""
    _validate_dim(d)
    return _cached_shift_gate(d).copy()


def clock_gate(d: int) -> qt.Qobj:
    """Generalized Z_d: Z_d|k> = omega^k |k>."""
    _validate_dim(d)
    return _cached_clock_gate(d).copy()


def chrestenson_gate(d: int) -> qt.Qobj:
    """Generalized Hadamard / QFT_d, (H_d)_{jk} = omega^{jk}/sqrt(d)."""
    _validate_dim(d)
    return _cached_chrestenson_gate(d).copy()


def phase_gate(d: int, k: int, m: int = 1) -> qt.Qobj:
    """Diagonal gate multiplying |k> by omega^m; identity on other basis states."""
    _validate_dim(d)
    if not (0 <= k < d):
        raise ValueError(f"k must satisfy 0 <= k < d={d}, got {k!r}")
    return _cached_phase_gate(d, k, m % d).copy()


# ---------------------------------------------------------------------------
# Public two-qudit gates
# ---------------------------------------------------------------------------

def csum_gate(d: int) -> qt.Qobj:
    """Generalized CNOT / controlled-sum: |a>|b> -> |a>|(a+b) mod d>."""
    _validate_dim(d)
    return _cached_csum_gate(d).copy()


def cz_gate(d: int) -> qt.Qobj:
    """Generalized CZ: |a>|b> -> omega^{a*b} |a>|b>."""
    _validate_dim(d)
    return _cached_cz_gate(d).copy()


def swap_gate(d: int) -> qt.Qobj:
    """Generalized SWAP: |a>|b> -> |b>|a>."""
    _validate_dim(d)
    return _cached_swap_gate(d).copy()


# ---------------------------------------------------------------------------
# Register embedding (what makes the library register-general)
# ---------------------------------------------------------------------------

def embed(op: qt.Qobj, positions: Union[int, Sequence[int]], m: int, d: int) -> qt.Qobj:
    """
    Embed a 1-qudit or 2-qudit operator into an m-qudit register at the
    given position(s), tensoring qt.qeye(d) at every other position.

    Parameters
    ----------
    op : qt.Qobj
        A d x d (single-qudit) or d**2 x d**2 (two-qudit) operator, e.g.
        the output of shift_gate(d), chrestenson_gate(d), csum_gate(d).
    positions : int or sequence of int
        - int: the register slot for a single-qudit `op`.
        - (control, target) pair: register slots for a two-qudit `op`
          (CSUM/CZ/SWAP). Positions need not be adjacent; the two-qudit
          op is reshaped internally to act on the two chosen slots with
          the rest of the register left untouched.
    m : int
        Total number of qudits in the register.
    d : int
        Qudit dimension.

    Returns
    -------
    qt.Qobj
        Operator on the full m-qudit register, dims == [[d]*m, [d]*m].
    """
    _validate_dim(d)
    if isinstance(positions, (int, np.integer)):
        pos_list = [int(positions)]
    else:
        pos_list = list(positions)

    for p in pos_list:
        _validate_position(p, m)

    n_local = len(pos_list)
    expected_dim = d ** n_local
    if op.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"op shape {op.shape} doesn't match a {n_local}-qudit operator "
            f"of dimension {d} (expected {(expected_dim, expected_dim)})"
        )

    if n_local == 1:
        ops = [qt.qeye(d)] * m
        ops[pos_list[0]] = op
        result = qt.tensor(ops)
        result.dims = _op_dims(d, m)
        return result

    if n_local == 2:
        # Build on identities everywhere except the two target slots, then
        # permute the tensor factors so the two-qudit op's own internal
        # ordering (row/col pair) lines up with (positions[0], positions[1])
        # regardless of how far apart they are in the register.
        remaining = [p for p in range(m) if p not in pos_list]
        ordered_positions = pos_list + remaining  # [p0, p1, rest...]
        ops = [op] + [qt.qeye(d)] * (m - 2)
        staged = qt.tensor(ops)  # acts on (p0, p1, *remaining) in that order
        staged.dims = _op_dims(d, m)

        # permute() reorders subsystems from `ordered_positions` layout
        # back to natural (0, 1, ..., m-1) layout.
        inverse_perm = np.argsort(ordered_positions)
        result = staged.permute(list(inverse_perm))
        result.dims = _op_dims(d, m)
        return result

    raise NotImplementedError("embed() currently supports 1- or 2-qudit operators only")


# ---------------------------------------------------------------------------
# Register-wide convenience builders
# ---------------------------------------------------------------------------

def chrestenson_all(m: int, d: int) -> qt.Qobj:
    """
    Chrestenson gate applied to every qudit in an m-qudit register:
    H_d^{\\otimes m}. Generalizes H^{\\otimes m} used to prepare the
    uniform superposition at the start of Grover.
    """
    _validate_dim(d)
    ops = [chrestenson_gate(d)] * m
    result = qt.tensor(ops)
    result.dims = _op_dims(d, m)
    return result


def identity(m: int, d: int) -> qt.Qobj:
    """Identity on an m-qudit register of dimension d."""
    _validate_dim(d)
    op = qt.qeye(d ** m)
    op.dims = _op_dims(d, m)
    return op
