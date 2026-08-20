"""
grover.py
=========
Ideal (noiseless) Grover search for a register of m qudits of dimension d.
Builds entirely on states.py and gates.py -- nothing here is hard-coded to
d=3 or to any particular register size, so the same functions run the
qutrit target case and the qubit baseline with no code duplication. Noise
is deliberately absent from this module; grover_noise.py (next file) will
reuse the same oracle/diffusion construction but propagate a density
matrix through Kraus channels between steps instead of a pure state.

Algorithm (unchanged in structure across d)
--------------------------------------------
1. Prepare |s> = H_d^{\\otimes m} |0...0>  (uniform_superposition, via
   gates.chrestenson_all).
2. Repeat `iterations` times:
     a. Oracle O: flip the sign of marked basis state(s), everything else
        untouched.  (The -1 phase-flip oracle is dimension-independent --
        Grover's amplitude-amplification argument only needs marked
        amplitudes to pick up a relative minus sign, regardless of d.)
     b. Diffusion D = 2|s><s| - I: inversion about the mean amplitude.
3. Measure -> success probability is the total weight on marked states.

Optimal iteration count generalizes directly from the qubit case:
    t_opt = floor( (pi/4) * sqrt(N / M) ),  N = d**m, M = #marked states.

Memory notes
------------
Oracle and diffusion are built as N x N operators (N = d**m), which is the
same asymptotic cost as any other m-qudit gate in this library -- fine at
the project's target register sizes (m up to ~6-7) but grows fast, so
nothing here silently loops calling this at larger m. Oracle is built as a
diagonal array directly (not via repeated embed() calls) since a diagonal
phase-flip has no local tensor structure to exploit; diffusion reuses
states.uniform_superposition and gates.identity rather than recomputing
either. Neither is cached -- unlike the small per-dimension gates in
gates.py, oracle/diffusion depend on (m, d, marked) jointly and are only
built once per run_grover() call, so caching would cost memory for no
reuse benefit at these problem sizes.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import qutip as qt

from .gates import _op_dims, chrestenson_all, identity
from .states import basis_state, uniform_superposition


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_dim(d: int) -> None:
    if not isinstance(d, (int, np.integer)) or d < 2:
        raise ValueError(f"Qudit dimension d must be an integer >= 2, got {d!r}")


def _normalize_marked(marked: Union[int, Sequence[int]], N: int) -> List[int]:
    """Accept a single flat index or a sequence of flat indices in [0, N)."""
    if isinstance(marked, (int, np.integer)):
        marked = [int(marked)]
    marked = sorted(set(int(x) for x in marked))
    if not marked:
        raise ValueError("marked must contain at least one index")
    for x in marked:
        if not (0 <= x < N):
            raise ValueError(f"marked index {x} out of range for N={N}")
    return marked


# ---------------------------------------------------------------------------
# Oracle and diffusion
# ---------------------------------------------------------------------------

def oracle(m: int, d: int, marked: Union[int, Sequence[int]]) -> qt.Qobj:
    """
    Phase-flip oracle on an m-qudit register: multiplies marked
    computational-basis state(s) by -1, leaves everything else untouched.
        O|x> = -|x>   if x is marked
        O|x> =  |x>   otherwise
    `marked` may be a single flat index in [0, d**m) or a sequence of them
    (multi-target search).
    """
    _validate_dim(d)
    N = d ** m
    marked_list = _normalize_marked(marked, N)
    diag = np.ones(N, dtype=complex)
    diag[marked_list] = -1.0
    return qt.Qobj(np.diag(diag), dims=_op_dims(d, m))


def diffusion_operator(m: int, d: int) -> qt.Qobj:
    """
    Grover diffusion operator on an m-qudit register: inversion about the
    mean amplitude, D = 2|s><s| - I, where |s> is the uniform
    superposition over all d**m basis states. Same construction regardless
    of d -- the only thing that changes with dimension is what |s> is.
    """
    _validate_dim(d)
    s = uniform_superposition(d, m)
    proj = s * s.dag()
    proj.dims = _op_dims(d, m)
    return 2 * proj - identity(m, d)


def optimal_iterations(N: int, M: int = 1) -> int:
    """
    Standard Grover iteration count, generalized to N = d**m search-space
    size and M marked items:
        t_opt = floor( (pi/4) * sqrt(N / M) )
    """
    if M <= 0 or M > N:
        raise ValueError(f"Need 1 <= M <= N, got M={M}, N={N}")
    return int(np.floor((np.pi / 4) * np.sqrt(N / M)))


# ---------------------------------------------------------------------------
# Running the algorithm
# ---------------------------------------------------------------------------

def run_grover(
    m: int,
    d: int,
    marked: Union[int, Sequence[int]],
    iterations: Optional[int] = None,
    return_history: bool = False,
) -> Tuple[qt.Qobj, int, Optional[List[qt.Qobj]]]:
    """
    Run ideal (noiseless) Grover search on an m-qudit register of
    dimension d, searching for `marked` basis state(s).

    Parameters
    ----------
    m : int
        Number of qudits in the register.
    d : int
        Qudit dimension (2 = qubit baseline, 3 = qutrit target case, ...).
    marked : int or sequence of int
        Flat index/indices (in [0, d**m)) of the marked state(s).
    iterations : int, optional
        Number of Grover iterations. Defaults to optimal_iterations(N, M).
    return_history : bool
        If True, also return the state after every iteration (including
        the initial uniform superposition at index 0), useful for
        plotting success probability vs. iteration count.

    Returns
    -------
    (final_state, iterations_used, history_or_None)
    """
    _validate_dim(d)
    N = d ** m
    marked_list = _normalize_marked(marked, N)
    M = len(marked_list)

    if iterations is None:
        iterations = optimal_iterations(N, M)

    psi = chrestenson_all(m, d) * basis_state(d, [0] * m, n=m)
    O = oracle(m, d, marked_list)
    D = diffusion_operator(m, d)

    history = [psi] if return_history else None
    for _ in range(iterations):
        psi = D * (O * psi)
        if return_history:
            history.append(psi)

    return psi, iterations, history


def success_probability(state: qt.Qobj, marked: Union[int, Sequence[int]]) -> float:
    """
    Total probability weight on the marked basis state(s) for a pure
    state. (grover_noise.py will provide the density-matrix analogue,
    since a mixed state's diagonal already gives populations directly.)
    """
    if not state.isket:
        raise ValueError("success_probability expects a ket; use the density-matrix "
                          "version in grover_noise.py for mixed states")
    N = state.shape[0]
    marked_list = _normalize_marked(marked, N)
    amps = state.full().flatten()
    return float(np.sum(np.abs(amps[marked_list]) ** 2))


def success_probability_trace(history: List[qt.Qobj], marked: Union[int, Sequence[int]]) -> List[float]:
    """Success probability at every step of a run_grover(..., return_history=True) history."""
    return [success_probability(psi, marked) for psi in history]
