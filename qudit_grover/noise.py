"""
noise.py
========
Physically anchored Kraus-channel noise model for a single qudit of
dimension d, built directly from the reference write-up: transmon
amplitude damping (bosonic sqrt(n)-enhanced cascade decay) and pure
dephasing, composed into one CPTP map applied once per gate layer.
Dimension-agnostic throughout -- built for the qutrit (d=3) target case,
but any d collapses correctly, and d=2 reduces exactly to the textbook
2-Kraus amplitude-damping / phase-damping channels.

Scope: this module acts on ONE qudit. grover_noise.py is responsible for
placing these single-qudit channels onto each register qudit and applying
them once per gate layer (oracle, diffusion, any intermediate gates).

Implemented (per the write-up's own verdicts):
    - Rate conversion (gamma, lambda from T1, Tphi).
    - Cascade amplitude damping, generalized to any d.
    - Pure dephasing, generalized to any d.
    - Bosonic T1 ladder (Gamma^(n->n-1) = n * Gamma^(1->0)) and a stated,
      not derived, Tphi ladder, generalized to any d.
    - Channel composition + completeness check (sum K_i^dag K_i = I).
    - Channel application: rho -> sum_i K_i rho K_i^dagger.
    - Quasiparticle poisoning, Level 1: stochastic per-shot T1 modulation.
    - mesolve cross-validation of the closed-form Kraus channel.

Deliberately NOT implemented here, matching the write-up's stated
assumptions: generalized/thermal amplitude damping (n_th negligible),
1/f (non-Markovian) dephasing, leakage beyond d, crosstalk, and a
dedicated QP Kraus channel (Level 2, future work). Readout/SPAM confusion
-matrix error is applied once at measurement time in simulation.py, not
per-gate here.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import qutip as qt


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_dim(d: int) -> None:
    if not isinstance(d, (int, np.integer)) or d < 2:
        raise ValueError(f"Qudit dimension d must be an integer >= 2, got {d!r}")


def _op_dims_single(d: int) -> List[list]:
    return [[d], [d]]


# ---------------------------------------------------------------------------
# Rate conversion (Eq. gamma_def / lambda_def)
# ---------------------------------------------------------------------------

def decay_probability(t: float, T1: float) -> float:
    """gamma(t; T1) = 1 - exp(-t / T1): population-decay probability over time t."""
    if T1 <= 0:
        raise ValueError(f"T1 must be positive, got {T1!r}")
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t!r}")
    return 1.0 - np.exp(-t / T1)


def dephasing_probability(t: float, Tphi: float) -> float:
    """lambda(t; Tphi) = 1 - exp(-t / Tphi): pure-dephasing probability over time t."""
    if Tphi <= 0:
        raise ValueError(f"Tphi must be positive, got {Tphi!r}")
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t!r}")
    return 1.0 - np.exp(-t / Tphi)


def Tphi_from_T1_T2(T1: float, T2: float) -> float:
    """
    Pure dephasing time from 1/T2 = 1/(2*T1) + 1/Tphi (Eq. t2relation).
    Using raw T2 inside a dephasing Kraus op double-counts the T1
    contribution already in the amplitude-damping channel -- this
    function is how you correctly extract Tphi from measured T1, T2.
    Raises if T2 > 2*T1 (unphysical: T2 <= 2*T1 always).
    """
    if T2 > 2 * T1:
        raise ValueError(f"T2={T2} exceeds the physical bound 2*T1={2*T1}.")
    inv_Tphi = 1.0 / T2 - 1.0 / (2.0 * T1)
    if inv_Tphi <= 0:
        raise ValueError("Computed 1/Tphi <= 0; check T1, T2 inputs.")
    return 1.0 / inv_Tphi


# ---------------------------------------------------------------------------
# Bosonic-enhanced transmon ladders (Eq. bosonic, generalized to any d)
# ---------------------------------------------------------------------------

def transmon_T1_ladder(T1_base: float, d: int) -> List[float]:
    """
    T1 for each downward transition n -> n-1, n = 1..d-1, from the
    bosonic sqrt(n) enhancement of transmon dipole matrix elements:
        Gamma_1^{(n -> n-1)} = n * Gamma_1^{(1 -> 0)}
        <=> T1^{(n -> n-1)} = T1_base / n
    (Eq. bosonic gives the d=3 case, n=2: T1^(2->1) = T1_base/2; this is
    the direct generalization to arbitrary d.) Returns
    [T1^(1->0), T1^(2->1), ..., T1^((d-1)->(d-2))].
    """
    _validate_dim(d)
    if T1_base <= 0:
        raise ValueError(f"T1_base must be positive, got {T1_base!r}")
    return [T1_base / n for n in range(1, d)]


def transmon_Tphi_ladder(Tphi_base: float, d: int, decay_ratio: float = 2.0) -> List[float]:
    """
    Tphi for each level's dephasing, n = 1..d-1. The write-up gives only
    Tphi^(1) (free) and Tphi^(2) ~ Tphi^(1)/2 to Tphi^(1)/3 as a *stated
    assumption*, not a derived scaling law like the T1 ladder -- there is
    no first-principles rule given for d > 3. This generalizes that
    assumption geometrically: Tphi^(n) = Tphi_base / (decay_ratio**(n-1)),
    so decay_ratio=2 reproduces the "~Tphi/2" end of the stated range and
    decay_ratio=3 the "~Tphi/3" end. Pass your own list directly to
    dephasing_kraus() if you have better per-level numbers.
    """
    _validate_dim(d)
    if Tphi_base <= 0:
        raise ValueError(f"Tphi_base must be positive, got {Tphi_base!r}")
    if decay_ratio <= 0:
        raise ValueError(f"decay_ratio must be positive, got {decay_ratio!r}")
    return [Tphi_base / (decay_ratio ** (n - 1)) for n in range(1, d)]


# ---------------------------------------------------------------------------
# Amplitude damping (cascade), Section AD, generalized to any d
# ---------------------------------------------------------------------------

def amplitude_damping_kraus(d: int, gammas: Sequence[float]) -> List[qt.Qobj]:
    """
    Cascade amplitude-damping Kraus set: only nearest-neighbor downward
    decay |n> -> |n-1> is allowed (dipole-forbidden direct jumps, as in a
    transmon), matching Section AD.

    gammas[n-1] = decay probability for transition n -> n-1, n = 1..d-1
    (gammas[0] is 1->0, gammas[1] is 2->1, ...). Typically
    transmon_T1_ladder() fed through decay_probability(t, .).

    Returns d Kraus operators:
        K_0 = diag(1, sqrt(1-gammas[0]), sqrt(1-gammas[1]), ...)
        K_n = sqrt(gammas[n-1]) |n-1><n|,  n = 1..d-1
    At d=2 this is exactly the textbook 2-Kraus amplitude-damping channel.
    """
    _validate_dim(d)
    if len(gammas) != d - 1:
        raise ValueError(f"Expected {d - 1} decay probabilities, got {len(gammas)}")
    for g in gammas:
        if not (0.0 <= g <= 1.0):
            raise ValueError(f"Decay probability must be in [0,1], got {g!r}")

    survival = [1.0] + [np.sqrt(1.0 - g) for g in gammas]
    K0 = qt.Qobj(np.diag(survival).astype(complex), dims=_op_dims_single(d))

    kraus_ops = [K0]
    for n in range(1, d):
        data = np.zeros((d, d), dtype=complex)
        data[n - 1, n] = np.sqrt(gammas[n - 1])
        kraus_ops.append(qt.Qobj(data, dims=_op_dims_single(d)))
    return kraus_ops


# ---------------------------------------------------------------------------
# Pure dephasing, generalized to any d
# ---------------------------------------------------------------------------

def dephasing_kraus(d: int, lambdas: Sequence[float]) -> List[qt.Qobj]:
    """
    Diagonal-only pure-dephasing Kraus set (no population transfer, only
    relative-phase randomization), generalized to any d.

    lambdas[n-1] = dephasing probability for level n, n = 1..d-1 (level 0
    is the phase reference). Typically transmon_Tphi_ladder() fed through
    dephasing_probability(t, .).

    Returns d Kraus operators:
        K_0 = diag(1, sqrt(1-lambdas[0]), sqrt(1-lambdas[1]), ...)
        K_n = diag with sqrt(lambdas[n-1]) at position n, zero elsewhere
    At d=2 this is exactly the textbook 2-Kraus phase-damping channel.
    """
    _validate_dim(d)
    if len(lambdas) != d - 1:
        raise ValueError(f"Expected {d - 1} dephasing probabilities, got {len(lambdas)}")
    for lam in lambdas:
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"Dephasing probability must be in [0,1], got {lam!r}")

    survival = [1.0] + [np.sqrt(1.0 - lam) for lam in lambdas]
    K0 = qt.Qobj(np.diag(survival).astype(complex), dims=_op_dims_single(d))

    kraus_ops = [K0]
    for n in range(1, d):
        diag = np.zeros(d, dtype=complex)
        diag[n] = np.sqrt(lambdas[n - 1])
        kraus_ops.append(qt.Qobj(np.diag(diag), dims=_op_dims_single(d)))
    return kraus_ops


# ---------------------------------------------------------------------------
# Composition, completeness, application
# ---------------------------------------------------------------------------

def compose_channels(kraus_a: Sequence[qt.Qobj], kraus_b: Sequence[qt.Qobj]) -> List[qt.Qobj]:
    """
    Compose two independent CPTP maps applied in sequence, b then a
    (E = E_a o E_b), via all pairwise Kraus products {K_i^a K_j^b}. For
    the write-up's dephasing-after-amplitude-damping composition:
        compose_channels(dephasing_kraus(...), amplitude_damping_kraus(...))
    Negligible product terms are NOT pruned here (kept exact); prune at
    the call site if performance demands it.
    """
    return [Ka * Kb for Ka in kraus_a for Kb in kraus_b]


def check_completeness(kraus_ops: Sequence[qt.Qobj], atol: float = 1e-9) -> bool:
    """
    Verify the CPTP completeness relation sum_i K_i^dagger K_i == I. Per
    the write-up: "always verify this numerically after any modification
    -- it is the single easiest thing to silently break."
    """
    if not kraus_ops:
        raise ValueError("kraus_ops is empty")
    d = kraus_ops[0].shape[0]
    total = sum(K.dag() * K for K in kraus_ops)
    ident = qt.qeye(d)
    ident.dims = kraus_ops[0].dims
    return (total - ident).norm() < atol


def apply_channel(rho: qt.Qobj, kraus_ops: Sequence[qt.Qobj]) -> qt.Qobj:
    """Apply a CPTP map to a density matrix: rho -> sum_i K_i rho K_i^dagger."""
    if not rho.isoper:
        raise ValueError("apply_channel expects a density matrix (operator), not a ket")
    result = sum(K * rho * K.dag() for K in kraus_ops)
    result.dims = rho.dims
    return result


# ---------------------------------------------------------------------------
# High-level convenience: one physically-anchored qudit channel per gate layer
# ---------------------------------------------------------------------------

def transmon_qudit_channel(
    d: int,
    T1: float,
    Tphi: float,
    t_g: float,
    Tphi_decay_ratio: float = 2.0,
) -> List[qt.Qobj]:
    """
    Build the composed (dephasing after amplitude damping) Kraus channel
    for a single d-level transmon qudit over one gate layer of duration
    t_g, from the write-up's reduced 2-free-parameter model ("Reduced,
    physically-anchored parameter set"): T1 and Tphi are the only free
    inputs, everything else is derived via transmon_T1_ladder /
    transmon_Tphi_ladder. Completeness is asserted at every stage.
    """
    _validate_dim(d)
    T1_ladder = transmon_T1_ladder(T1, d)
    Tphi_ladder = transmon_Tphi_ladder(Tphi, d, decay_ratio=Tphi_decay_ratio)

    gammas = [decay_probability(t_g, t1) for t1 in T1_ladder]
    lambdas = [dephasing_probability(t_g, tphi) for tphi in Tphi_ladder]

    K_AD = amplitude_damping_kraus(d, gammas)
    K_phi = dephasing_kraus(d, lambdas)

    assert check_completeness(K_AD), "amplitude damping Kraus set failed completeness"
    assert check_completeness(K_phi), "dephasing Kraus set failed completeness"

    composed = compose_channels(K_phi, K_AD)
    assert check_completeness(composed), "composed channel failed completeness"
    return composed


# ---------------------------------------------------------------------------
# Quasiparticle poisoning -- Level 1 (stochastic T1 modulation)
# ---------------------------------------------------------------------------

def sample_effective_T1(
    T1_base: float,
    p_qp: float,
    Gamma_qp: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    One Monte Carlo sample of the effective T1 under Level-1 quasiparticle
    poisoning: with probability p_qp, add a poisoning rate Gamma_qp to the
    baseline decay rate for this shot/layer:
        1/T1_eff = 1/T1_base + x_qp * Gamma_qp,  x_qp ~ Bernoulli(p_qp)
    Feed T1_eff into transmon_T1_ladder()/decay_probability() in place of
    a fixed T1 inside simulation.py's Monte Carlo loop.
    """
    if not (0.0 <= p_qp <= 1.0):
        raise ValueError(f"p_qp must be in [0,1], got {p_qp!r}")
    if Gamma_qp < 0:
        raise ValueError(f"Gamma_qp must be non-negative, got {Gamma_qp!r}")
    rng = rng if rng is not None else np.random.default_rng()
    x_qp = rng.random() < p_qp
    inv_T1_eff = 1.0 / T1_base + (Gamma_qp if x_qp else 0.0)
    return 1.0 / inv_T1_eff


# ---------------------------------------------------------------------------
# Validation against qutip.mesolve (Section validation)
# ---------------------------------------------------------------------------

def validate_amplitude_damping_vs_mesolve(
    d: int,
    T1_ladder: Sequence[float],
    t: float,
    initial_state: Optional[qt.Qobj] = None,
    n_substeps: int = 200,
) -> float:
    """
    Cross-check the closed-form cascade Kraus amplitude-damping channel
    against direct Lindblad integration (qutip.mesolve), per Section
    "Numerical Validation". Builds Lindblad jump operators
    L_n = sqrt(Gamma_n) |n-1><n| with Gamma_n = 1/T1_ladder[n-1],
    integrates with zero Hamiltonian over total time t, and separately
    applies the closed-form Kraus channel (exact for any t, not just
    infinitesimal steps -- n_substeps only controls mesolve's own
    numerical resolution). Returns the trace distance between the two
    final density matrices (should be small; use as a sanity test, not
    part of the main simulation loop).
    """
    _validate_dim(d)
    if len(T1_ladder) != d - 1:
        raise ValueError(f"Expected {d - 1} T1 values, got {len(T1_ladder)}")

    if initial_state is None:
        # Start in the top level -- decays through the full cascade.
        ket = qt.basis(d, d - 1)
        rho0 = ket * ket.dag()
    else:
        rho0 = initial_state if initial_state.isoper else initial_state * initial_state.dag()
    rho0.dims = _op_dims_single(d)

    c_ops = []
    for n in range(1, d):
        Gamma_n = 1.0 / T1_ladder[n - 1]
        data = np.zeros((d, d), dtype=complex)
        data[n - 1, n] = np.sqrt(Gamma_n)
        c_ops.append(qt.Qobj(data, dims=_op_dims_single(d)))
    H0 = qt.Qobj(np.zeros((d, d)), dims=_op_dims_single(d))
    tlist = np.linspace(0, t, n_substeps)
    result = qt.mesolve(H0, rho0, tlist, c_ops=c_ops)
    rho_mesolve = result.states[-1]
    rho_mesolve.dims = _op_dims_single(d)

    gammas = [decay_probability(t, t1) for t1 in T1_ladder]
    K_AD = amplitude_damping_kraus(d, gammas)
    rho_kraus = apply_channel(rho0, K_AD)

    return qt.tracedist(rho_mesolve, rho_kraus)
