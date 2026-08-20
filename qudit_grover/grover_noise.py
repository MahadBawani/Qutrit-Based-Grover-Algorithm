"""
grover_noise.py
================
Noisy Grover search for a register of m qudits of dimension d: reuses the
exact oracle/diffusion construction from grover.py, but propagates a
density matrix through the transmon Kraus channels in noise.py between
gate layers instead of a pure state.

Noise placement
----------------
noise.py builds a physically-anchored CPTP channel for ONE qudit over one
gate layer (amplitude damping + dephasing, composed). This module is
where that gets placed onto the register: after every unitary layer
(oracle, then diffusion), the single-qudit channel is applied to EACH of
the m register qudits, via gates.embed() -- the same register-placement
primitive the rest of this library uses for gates, so a noise Kraus
operator is embedded exactly the way a shift/clock/Chrestenson gate would
be.

Independent single-qudit channels commute, so instead of building one
combined d**m-sized Kraus set (the tensor product over all m qudits, each
contributing d Kraus operators -> d**m operators total), the m
applications are done sequentially: embed the local d-operator Kraus set
onto qudit i via gates.embed(K, i, m, d), apply it, move to the next
qudit. Same physical channel, same result, m * d operators touched per
layer instead of d**m.

Two gate layers per Grover iteration (oracle, diffusion) => two noise
applications per iteration, each touching all m qudits.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import qutip as qt

from .gates import _op_dims, chrestenson_all, embed
from .states import basis_state
from .grover import (
    _normalize_marked,
    _validate_dim,
    diffusion_operator,
    oracle,
    optimal_iterations,
)
from .noise import apply_channel, transmon_qudit_channel


# ---------------------------------------------------------------------------
# Placing a single-qudit channel onto every qudit in the register
# ---------------------------------------------------------------------------

def embed_channel(kraus_ops: Sequence[qt.Qobj], position: int, m: int, d: int) -> List[qt.Qobj]:
    """
    Embed every Kraus operator of a single-qudit channel onto register
    slot `position`, via gates.embed -- identity on every other qudit,
    same placement convention as any other single-qudit gate in this
    library.
    """
    return [embed(K, position, m, d) for K in kraus_ops]


def apply_register_noise(
    rho: qt.Qobj,
    m: int,
    d: int,
    T1: float,
    Tphi: float,
    t_g: float,
    Tphi_decay_ratio: float = 2.0,
) -> qt.Qobj:
    """
    Apply the transmon_qudit_channel (amplitude damping + dephasing,
    composed) to every qudit in the register, one qudit at a time. Since
    the per-qudit channels are independent, applying them sequentially
    gives the same result as the full tensor-product channel while only
    touching m * d Kraus operators per call instead of d**m. The local
    channel itself is built once per call and reused across all m qudits
    (identical noise parameters per qudit) -- pass per-qudit T1/Tphi in a
    loop at the call site if the register is heterogeneous.
    """
    _validate_dim(d)
    kraus_local = transmon_qudit_channel(d, T1, Tphi, t_g, Tphi_decay_ratio)
    for i in range(m):
        rho = apply_channel(rho, embed_channel(kraus_local, i, m, d))
    return rho


# ---------------------------------------------------------------------------
# Running the noisy algorithm
# ---------------------------------------------------------------------------

def run_grover_noisy(
    m: int,
    d: int,
    marked: Union[int, Sequence[int]],
    T1: float,
    Tphi: float,
    t_g: float,
    iterations: Optional[int] = None,
    Tphi_decay_ratio: float = 2.0,
    return_history: bool = False,
) -> Tuple[qt.Qobj, int, Optional[List[qt.Qobj]]]:
    """
    Run noisy Grover search on an m-qudit register of dimension d,
    searching for `marked` basis state(s), with the transmon noise
    channel from noise.py applied to every qudit after every gate layer.

    Parameters
    ----------
    m, d, marked, iterations : same as grover.run_grover.
    T1 : float
        Base T1 (amplitude-damping time) for the 1->0 transition, fed
        through transmon_T1_ladder inside transmon_qudit_channel.
    Tphi : float
        Base Tphi (pure-dephasing time) for level 1, fed through
        transmon_Tphi_ladder inside transmon_qudit_channel.
    t_g : float
        Duration of one gate layer (same units as T1, Tphi). Applied once
        after the oracle and once after the diffusion operator, i.e. two
        noise applications per Grover iteration.
    Tphi_decay_ratio : float
        Passed straight through to transmon_qudit_channel's
        transmon_Tphi_ladder (see noise.py for what this controls).
    return_history : bool
        If True, also return the density matrix after every iteration
        (including the initial, noise-free uniform superposition at
        index 0), mirroring grover.run_grover's return_history.

    Returns
    -------
    (final_rho, iterations_used, history_or_None)
    """
    _validate_dim(d)
    N = d ** m
    marked_list = _normalize_marked(marked, N)
    M = len(marked_list)

    if iterations is None:
        iterations = optimal_iterations(N, M)

    psi0 = chrestenson_all(m, d) * basis_state(d, [0] * m, n=m)
    rho = psi0 * psi0.dag()
    rho.dims = _op_dims(d, m)

    O = oracle(m, d, marked_list)
    D = diffusion_operator(m, d)

    history = [rho] if return_history else None
    for _ in range(iterations):
        rho = O * rho * O.dag()
        rho = apply_register_noise(rho, m, d, T1, Tphi, t_g, Tphi_decay_ratio)

        rho = D * rho * D.dag()
        rho = apply_register_noise(rho, m, d, T1, Tphi, t_g, Tphi_decay_ratio)

        if return_history:
            history.append(rho)

    return rho, iterations, history


# ---------------------------------------------------------------------------
# Density-matrix analogues of grover.py's success-probability helpers
# ---------------------------------------------------------------------------

def success_probability_mixed(rho: qt.Qobj, marked: Union[int, Sequence[int]]) -> float:
    """
    Total probability weight on the marked basis state(s) for a mixed
    state -- density-matrix analogue of grover.success_probability. A
    mixed state's populations are read straight off the diagonal, no
    amplitude-squaring needed.
    """
    if not rho.isoper:
        raise ValueError("success_probability_mixed expects a density matrix; use "
                          "grover.success_probability for a pure ket")
    N = rho.shape[0]
    marked_list = _normalize_marked(marked, N)
    diag = np.real(rho.full().diagonal())
    return float(np.sum(diag[marked_list]))


def success_probability_trace_mixed(
    history: List[qt.Qobj], marked: Union[int, Sequence[int]]
) -> List[float]:
    """Success probability at every step of a run_grover_noisy(..., return_history=True) history."""
    return [success_probability_mixed(rho, marked) for rho in history]
