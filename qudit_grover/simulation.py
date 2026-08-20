"""
simulation.py
==============
Final benchmarking script for the qudit Grover project. Runs every ideal
and noisy Grover simulation used in the report, computes the fidelity /
purity / trace-distance / coherence benchmark suite, and writes out the
CSVs and figures.

This is deliberately the "hardcoded" layer of the project: every sweep
range, marked index, and reference noise parameter below is a concrete
choice, not a generic library function. states.py / gates.py / grover.py
/ noise.py / grover_noise.py stay dimension-general; this file spends
that generality on one fixed, defensible comparison.

The central scientific question (NOT decided in advance -- see
`write_summary()` at the bottom, which reports what the numbers actually
show, computed after the fact)
--------------------------------------------------------------------------
At matched search-space size (N=9 qutrit vs N=8 qubit) and matched
iteration count (both land on t_opt = 2, verified numerically below --
this match is not a coincidence, it is why N=8/9 was chosen as the
canonical comparison point throughout this file), the qutrit register
uses fewer physical carriers (m=2 vs m=3) and a shallower circuit, but
each physical qudit carries an extra leakage/damping channel (level |2>)
and decays faster on that channel (bosonic sqrt(n) enhancement,
T1^(2->1) = T1/2). Three questions, answered by the data below rather
than assumed:
    Q1: Does Ps increase through Grover amplification at all (sanity
        check on the whole pipeline)?                    -> Figure 1
    Q2: Does the qutrit provide a larger search space per physical
        carrier (3^m > 2^m)?                              -> Figure 2
    Q3: Does that larger-Hilbert-space advantage SURVIVE decoherence, or
        does the extra damping channel erase it at realistic T1/Tphi?
        -> Figures 3-7, Table 2. This is the interesting, open question;
        the write-up should report whichever answer the numbers give.

Outputs (all paths relative to the current working directory)
--------------------------------------------------------------------------
results/ideal_grover.csv          Table 1 (ideal half): d, m, N, t_opt, Ps
results/noisy_grover.csv          Table 1 (noisy half): + noisy Ps, fidelity
results/T1_sweep.csv              Figure 4 + Table 2 raw data
results/Tphi_sweep.csv            Figure 5 raw data
results/matched_T1_Tphi_sweep.csv Table 2 raw data (T1 = Tphi diagonal sweep)
results/validation.csv            noise.py's mesolve cross-check, tabulated
results/summary_table1.md         Table 1, rendered as markdown
results/summary_table2.md         Table 2, rendered as markdown
results/summary.md                Q1/Q2/Q3 findings, written from the data

figures/fig1_ideal_grover.png     Ps vs iteration, d=2 vs d=3 (matched N)
figures/fig2_scaling.png          t_opt vs N, both dimensions
figures/fig3_noise.png            4 curves: qubit/qutrit x ideal/noisy
figures/fig4_T1.png               Ps vs T1, qubit vs qutrit
figures/fig5_Tphi.png             Ps vs Tphi, qubit vs qutrit
figures/fig6_qubit_heatmap.png    T1-Tphi heatmap, qubit
figures/fig7_qutrit_heatmap.png   T1-Tphi heatmap, qutrit
figures/fig8_purity.png           optional: purity vs iteration
figures/fig9_fidelity.png         optional: fidelity vs iteration
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import qutip as qt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .grover import optimal_iterations, run_grover, success_probability, success_probability_trace
from .grover_noise import (
    run_grover_noisy,
    success_probability_mixed,
    success_probability_trace_mixed,
)
from .noise import transmon_T1_ladder, validate_amplitude_damping_vs_mesolve

# ---------------------------------------------------------------------------
# Fixed configuration (the "hardcoded" part of the project)
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

# Reference transmon parameters (see noise.py's write-up, Section: Reference
# parameter table). t_g here is a per-*layer* duration (oracle or diffusion
# as a whole), not a single physical gate time -- run_grover_noisy applies
# one noise channel per layer, so t_g is the effective duration of
# everything that happens in that layer. Chosen near the shorter end of
# the two-qutrit gate time range (100-300 ns) since a diffusion/oracle
# layer on a small register is a handful of single- and two-qudit gates.
DEFAULT_T1_US = 50.0
DEFAULT_TPHI_US = 20.0
DEFAULT_TG_US = 0.05  # 50 ns

# The canonical matched comparison: qubit m=3 (N=8) vs qutrit m=2 (N=9).
# Confirmed numerically (not assumed) to share t_opt=2 -- see
# verify_matched_case() below, called at the start of main().
QUBIT_M_MATCHED = 3
QUTRIT_M_MATCHED = 2
MARKED_INDEX = 0  # flat index of the marked state; arbitrary but fixed

# Table 1 sweep ranges
QUBIT_MS = (1, 2, 3, 4)
QUTRIT_MS = (1, 2, 3, 4)

# Figure 2 scaling sweep (wider, no noise simulation needed -- iteration
# count is a closed-form function of N, so this can go further than the
# density-matrix sweeps above without a runtime cost).
QUBIT_MS_SCALING = tuple(range(1, 9))
QUTRIT_MS_SCALING = tuple(range(1, 7))

# T1 / Tphi sweep grids (microseconds), log-spaced across the literature
# range from noise.py's reference table (T1: 30-100 us) extended a bit on
# both ends to see the channel's full effect.
T1_SWEEP_US = np.geomspace(2, 300, 14)
TPHI_SWEEP_US = np.geomspace(2, 300, 14)

# Heatmap grid (coarser -- each point is a full run_grover_noisy call)
HEATMAP_T1_US = np.geomspace(5, 200, 8)
HEATMAP_TPHI_US = np.geomspace(5, 200, 8)

# Table 2's matched diagonal sweep
MATCHED_DIAGONAL_US = (10.0, 20.0, 40.0, 80.0, 160.0)

# Default readout confusion matrices (columns = true state, rows =
# measured state; M @ p_true = p_measured), representative transmon
# readout fidelities. Applied once at measurement per noise.py's write-up
# (Section: Readout / SPAM error), not per gate layer.
QUBIT_CONFUSION = np.array([
    [0.98, 0.05],
    [0.02, 0.95],
])
QUTRIT_CONFUSION = np.array([
    [0.97, 0.02, 0.01],
    [0.02, 0.95, 0.04],
    [0.01, 0.03, 0.95],
])


def _ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def verify_matched_case() -> None:
    """
    Confirm numerically (not assume) that qubit m=3 (N=8) and qutrit m=2
    (N=9) share the same optimal iteration count, which is the entire
    premise of using them as the canonical matched comparison throughout
    this file. Raises if that stops being true (e.g. if MARKED_INDEX
    logic or optimal_iterations() changes upstream).
    """
    N_qubit = 2 ** QUBIT_M_MATCHED
    N_qutrit = 3 ** QUTRIT_M_MATCHED
    t_qubit = optimal_iterations(N_qubit, 1)
    t_qutrit = optimal_iterations(N_qutrit, 1)
    if t_qubit != t_qutrit:
        raise AssertionError(
            f"Matched-case assumption broken: qubit N={N_qubit} gives "
            f"t_opt={t_qubit}, qutrit N={N_qutrit} gives t_opt={t_qutrit}. "
            f"Update QUBIT_M_MATCHED/QUTRIT_M_MATCHED or the surrounding "
            f"figures that assume a shared iteration count."
        )
    print(f"[verify_matched_case] qubit N={N_qubit} t_opt={t_qubit}, "
          f"qutrit N={N_qutrit} t_opt={t_qutrit} -- match confirmed")


# ---------------------------------------------------------------------------
# Benchmark metrics (fidelity, purity, trace distance, coherence, readout)
# ---------------------------------------------------------------------------

def fidelity_to_pure(rho: qt.Qobj, psi_ideal: qt.Qobj) -> float:
    """
    F = <psi_ideal| rho |psi_ideal>, the pure-target simplification of
    Uhlmann-Jozsa fidelity (see the fidelity/benchmarks reference doc).
    NOTE: this is NOT the same convention as qutip's own qt.fidelity(),
    which returns the square root of this quantity (Nielsen & Chuang
    convention) -- verified numerically against this qutip version before
    writing this function. Using the un-square-rooted overlap here to
    match the reference doc exactly; do not mix the two conventions when
    reporting numbers.
    """
    if not psi_ideal.isket:
        raise ValueError("psi_ideal must be a ket")
    overlap = psi_ideal.dag() * rho * psi_ideal
    return float(np.real(complex(overlap)))


def purity_of(rho: qt.Qobj) -> float:
    """Tr(rho^2)."""
    return float(np.real((rho * rho).tr()))


def trace_distance(rho: qt.Qobj, sigma: qt.Qobj) -> float:
    """(1/2) * sum_i |eigenvalues of (rho - sigma)| -- qutip's own tracedist."""
    return float(qt.tracedist(rho, sigma))


def coherence_l1(rho: qt.Qobj) -> float:
    """l1-norm of coherence: sum of |off-diagonal element| in the computational basis."""
    arr = rho.full()
    off_diag = arr - np.diag(np.diag(arr))
    return float(np.sum(np.abs(off_diag)))


def confusion_matrix_for(d: int) -> np.ndarray:
    if d == 2:
        return QUBIT_CONFUSION.copy()
    if d == 3:
        return QUTRIT_CONFUSION.copy()
    raise ValueError(f"No default confusion matrix defined for d={d}; pass one explicitly.")


def register_confusion_matrix(confusion_single: np.ndarray, m: int) -> np.ndarray:
    """Per-qudit readout errors are independent -> register confusion matrix is M^{⊗m}."""
    M = confusion_single
    for _ in range(m - 1):
        M = np.kron(M, confusion_single)
    return M


def apply_readout_error(
    rho: qt.Qobj, m: int, d: int, confusion_single: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Classical post-processing step applied once at measurement (NOT a
    channel on rho itself -- see noise.py's write-up, Section: Readout /
    SPAM error): perturbs the measured probability distribution over the
    d**m register outcomes by the per-qudit confusion matrix, tensored
    across the register. Returns the corrected probability vector.
    """
    if confusion_single is None:
        confusion_single = confusion_matrix_for(d)
    M_reg = register_confusion_matrix(confusion_single, m)
    probs_true = np.real(rho.full().diagonal())
    probs_measured = M_reg @ probs_true
    probs_measured = np.clip(probs_measured, 0, None)
    probs_measured = probs_measured / probs_measured.sum()
    return probs_measured


def success_from_probs(probs: np.ndarray, marked: Sequence[int]) -> float:
    return float(np.sum(probs[list(marked)]))


# ---------------------------------------------------------------------------
# Table 1: ideal_grover.csv + noisy_grover.csv
# ---------------------------------------------------------------------------

def compute_ideal_table(
    qubit_ms: Sequence[int] = QUBIT_MS, qutrit_ms: Sequence[int] = QUTRIT_MS
) -> List[Dict]:
    rows = []
    for d, ms, label in [(2, qubit_ms, "Qubit"), (3, qutrit_ms, "Qutrit")]:
        for m in ms:
            N = d ** m
            psi, t_opt, _ = run_grover(m, d, marked=MARKED_INDEX)
            ps = success_probability(psi, MARKED_INDEX)
            rows.append({
                "system": label, "d": d, "m": m, "N": N,
                "t_opt": t_opt, "ideal_Ps": ps,
            })
    return rows


def compute_noisy_table(
    qubit_ms: Sequence[int] = QUBIT_MS,
    qutrit_ms: Sequence[int] = QUTRIT_MS,
    T1: float = DEFAULT_T1_US,
    Tphi: float = DEFAULT_TPHI_US,
    t_g: float = DEFAULT_TG_US,
) -> List[Dict]:
    rows = []
    for d, ms, label in [(2, qubit_ms, "Qubit"), (3, qutrit_ms, "Qutrit")]:
        for m in ms:
            N = d ** m
            rho, t_opt, _ = run_grover_noisy(m, d, MARKED_INDEX, T1, Tphi, t_g)
            ps_noisy = success_probability_mixed(rho, MARKED_INDEX)
            psi_ideal, _, _ = run_grover(m, d, MARKED_INDEX, iterations=t_opt)
            fid = fidelity_to_pure(rho, psi_ideal)
            pur = purity_of(rho)
            rows.append({
                "system": label, "d": d, "m": m, "N": N, "t_opt": t_opt,
                "noisy_Ps": ps_noisy, "fidelity": fid, "purity": pur,
            })
    return rows


# ---------------------------------------------------------------------------
# T1 / Tphi sweeps (Figures 4, 5, matched-case data for Figure 3, 8, 9)
# ---------------------------------------------------------------------------

def sweep_T1(
    T1_values: np.ndarray = T1_SWEEP_US,
    Tphi_fixed: float = DEFAULT_TPHI_US,
    t_g: float = DEFAULT_TG_US,
) -> List[Dict]:
    rows = []
    for T1 in T1_values:
        for d, m, label in [(2, QUBIT_M_MATCHED, "Qubit"), (3, QUTRIT_M_MATCHED, "Qutrit")]:
            rho, t_opt, _ = run_grover_noisy(m, d, MARKED_INDEX, float(T1), Tphi_fixed, t_g)
            ps = success_probability_mixed(rho, MARKED_INDEX)
            rows.append({"system": label, "d": d, "m": m, "T1_us": float(T1),
                         "Tphi_us": Tphi_fixed, "t_opt": t_opt, "Ps": ps})
    return rows


def sweep_Tphi(
    Tphi_values: np.ndarray = TPHI_SWEEP_US,
    T1_fixed: float = DEFAULT_T1_US,
    t_g: float = DEFAULT_TG_US,
) -> List[Dict]:
    rows = []
    for Tphi in Tphi_values:
        for d, m, label in [(2, QUBIT_M_MATCHED, "Qubit"), (3, QUTRIT_M_MATCHED, "Qutrit")]:
            rho, t_opt, _ = run_grover_noisy(m, d, MARKED_INDEX, T1_fixed, float(Tphi), t_g)
            ps = success_probability_mixed(rho, MARKED_INDEX)
            rows.append({"system": label, "d": d, "m": m, "T1_us": T1_fixed,
                         "Tphi_us": float(Tphi), "t_opt": t_opt, "Ps": ps})
    return rows


def sweep_matched_diagonal(
    values_us: Sequence[float] = MATCHED_DIAGONAL_US, t_g: float = DEFAULT_TG_US
) -> List[Dict]:
    """Table 2: T1 = Tphi, swept together, qubit vs qutrit at the matched N=8/9 case."""
    rows = []
    for val in values_us:
        rho_q, _, _ = run_grover_noisy(QUBIT_M_MATCHED, 2, MARKED_INDEX, val, val, t_g)
        rho_t, _, _ = run_grover_noisy(QUTRIT_M_MATCHED, 3, MARKED_INDEX, val, val, t_g)
        ps_qubit = success_probability_mixed(rho_q, MARKED_INDEX)
        ps_qutrit = success_probability_mixed(rho_t, MARKED_INDEX)
        rows.append({
            "T1_us": val, "Tphi_us": val,
            "Ps_qubit": ps_qubit, "Ps_qutrit": ps_qutrit,
            "qutrit_advantage": ps_qutrit - ps_qubit,
        })
    return rows


# ---------------------------------------------------------------------------
# Heatmaps (Figures 6, 7)
# ---------------------------------------------------------------------------

def compute_heatmap(
    d: int, m: int, T1_grid: np.ndarray = HEATMAP_T1_US, Tphi_grid: np.ndarray = HEATMAP_TPHI_US,
    t_g: float = DEFAULT_TG_US,
) -> np.ndarray:
    """Returns a Ps grid of shape (len(Tphi_grid), len(T1_grid)) -- rows=Tphi, cols=T1."""
    grid = np.zeros((len(Tphi_grid), len(T1_grid)))
    for i, tphi in enumerate(Tphi_grid):
        for j, t1 in enumerate(T1_grid):
            rho, _, _ = run_grover_noisy(m, d, MARKED_INDEX, float(t1), float(tphi), t_g)
            grid[i, j] = success_probability_mixed(rho, MARKED_INDEX)
    return grid


# ---------------------------------------------------------------------------
# Validation table (noise.py's mesolve cross-check)
# ---------------------------------------------------------------------------

def compute_validation_table() -> List[Dict]:
    rows = []
    for d in (2, 3):
        for T1 in (20.0, 50.0, 100.0):
            for t_g in (0.03, 0.05, 0.1):
                ladder = transmon_T1_ladder(T1, d)
                dist = validate_amplitude_damping_vs_mesolve(d, ladder, t=t_g)
                rows.append({"d": d, "T1_us": T1, "t_g_us": t_g, "trace_distance": dist})
    return rows


# ---------------------------------------------------------------------------
# CSV writer (no pandas dependency needed for the schema-fixed tables, but
# pandas is used here since it's already a dependency of the plotting env
# and gives cleaner CSV formatting for mixed int/float columns)
# ---------------------------------------------------------------------------

def _write_csv(rows: List[Dict], path: str) -> None:
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig1_ideal_grover() -> None:
    _, t_qubit, hist_qubit = run_grover(QUBIT_M_MATCHED, 2, MARKED_INDEX, return_history=True)
    _, t_qutrit, hist_qutrit = run_grover(QUTRIT_M_MATCHED, 3, MARKED_INDEX, return_history=True)
    trace_qubit = success_probability_trace(hist_qubit, MARKED_INDEX)
    trace_qutrit = success_probability_trace(hist_qutrit, MARKED_INDEX)

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(trace_qubit)), trace_qubit, "o-", label=f"Qubit (d=2, m={QUBIT_M_MATCHED}, N=8)")
    plt.plot(range(len(trace_qutrit)), trace_qutrit, "s-", label=f"Qutrit (d=3, m={QUTRIT_M_MATCHED}, N=9)")
    plt.xlabel("Grover iteration")
    plt.ylabel("Success probability $P_s$")
    plt.title("Ideal Grover search: $P_s$ vs iteration (matched N)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_ideal_grover.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  wrote {path}")


def fig2_scaling() -> None:
    Ns_qubit = [2 ** m for m in QUBIT_MS_SCALING]
    Ns_qutrit = [3 ** m for m in QUTRIT_MS_SCALING]
    t_qubit = [optimal_iterations(N, 1) for N in Ns_qubit]
    t_qutrit = [optimal_iterations(N, 1) for N in Ns_qutrit]

    N_theory = np.geomspace(2, max(Ns_qubit + Ns_qutrit), 200)
    t_theory = (np.pi / 4) * np.sqrt(N_theory)

    plt.figure(figsize=(6, 4))
    plt.plot(N_theory, t_theory, "--", color="gray", label=r"theory: $(\pi/4)\sqrt{N}$")
    plt.plot(Ns_qubit, t_qubit, "o", label="Qubit (d=2)")
    plt.plot(Ns_qutrit, t_qutrit, "s", label="Qutrit (d=3)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Search space size $N = d^m$")
    plt.ylabel(r"Optimal iterations $t_{\rm opt}$")
    plt.title(r"Grover iteration scaling: $t_{\rm opt}$ vs $N$")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_scaling.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  wrote {path}")


def fig3_noise() -> None:
    _, _, hist_qubit_ideal = run_grover(QUBIT_M_MATCHED, 2, MARKED_INDEX, return_history=True)
    _, _, hist_qutrit_ideal = run_grover(QUTRIT_M_MATCHED, 3, MARKED_INDEX, return_history=True)
    _, _, hist_qubit_noisy = run_grover_noisy(
        QUBIT_M_MATCHED, 2, MARKED_INDEX, DEFAULT_T1_US, DEFAULT_TPHI_US, DEFAULT_TG_US,
        return_history=True,
    )
    _, _, hist_qutrit_noisy = run_grover_noisy(
        QUTRIT_M_MATCHED, 3, MARKED_INDEX, DEFAULT_T1_US, DEFAULT_TPHI_US, DEFAULT_TG_US,
        return_history=True,
    )

    trace_qubit_ideal = success_probability_trace(hist_qubit_ideal, MARKED_INDEX)
    trace_qutrit_ideal = success_probability_trace(hist_qutrit_ideal, MARKED_INDEX)
    trace_qubit_noisy = success_probability_trace_mixed(hist_qubit_noisy, MARKED_INDEX)
    trace_qutrit_noisy = success_probability_trace_mixed(hist_qutrit_noisy, MARKED_INDEX)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(trace_qubit_ideal, "o--", color="tab:blue", label="Qubit, ideal")
    plt.plot(trace_qubit_noisy, "o-", color="tab:blue", alpha=0.7, label="Qubit, noisy")
    plt.plot(trace_qutrit_ideal, "s--", color="tab:orange", label="Qutrit, ideal")
    plt.plot(trace_qutrit_noisy, "s-", color="tab:orange", alpha=0.7, label="Qutrit, noisy")
    plt.xlabel("Grover iteration")
    plt.ylabel("Success probability $P_s$")
    plt.title(f"Ideal vs noisy Grover (T1={DEFAULT_T1_US}\u03bcs, "
              f"T\u03c6={DEFAULT_TPHI_US}\u03bcs, t_g={DEFAULT_TG_US}\u03bcs)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_noise.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  wrote {path}")


def fig4_T1(rows_T1: List[Dict]) -> None:
    qubit = [r for r in rows_T1 if r["system"] == "Qubit"]
    qutrit = [r for r in rows_T1 if r["system"] == "Qutrit"]
    plt.figure(figsize=(6, 4))
    plt.plot([r["T1_us"] for r in qubit], [r["Ps"] for r in qubit], "o-", label="Qubit (N=8)")
    plt.plot([r["T1_us"] for r in qutrit], [r["Ps"] for r in qutrit], "s-", label="Qutrit (N=9)")
    plt.xscale("log")
    plt.xlabel(r"$T_1$ ($\mu$s)")
    plt.ylabel("Success probability $P_s$ at $t_{\\rm opt}$")
    plt.title(f"$P_s$ vs $T_1$ (T\u03c6 fixed at {DEFAULT_TPHI_US}\u03bcs)")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_T1.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  wrote {path}")


def fig5_Tphi(rows_Tphi: List[Dict]) -> None:
    qubit = [r for r in rows_Tphi if r["system"] == "Qubit"]
    qutrit = [r for r in rows_Tphi if r["system"] == "Qutrit"]
    plt.figure(figsize=(6, 4))
    plt.plot([r["Tphi_us"] for r in qubit], [r["Ps"] for r in qubit], "o-", label="Qubit (N=8)")
    plt.plot([r["Tphi_us"] for r in qutrit], [r["Ps"] for r in qutrit], "s-", label="Qutrit (N=9)")
    plt.xscale("log")
    plt.xlabel(r"$T_\phi$ ($\mu$s)")
    plt.ylabel("Success probability $P_s$ at $t_{\\rm opt}$")
    plt.title(f"$P_s$ vs T\u03c6 ($T_1$ fixed at {DEFAULT_T1_US}\u03bcs)")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_Tphi.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  wrote {path}")


def fig6_and_fig7_heatmaps() -> None:
    grid_qubit = compute_heatmap(2, QUBIT_M_MATCHED)
    grid_qutrit = compute_heatmap(3, QUTRIT_M_MATCHED)

    # Shared color scale across both heatmaps, sized to the actual data
    # range (not a fixed [0,1]) -- at these small N/iteration counts the
    # noise-induced variation is a matter of a few percent, and a [0,1]
    # scale would wash that out to a solid block of color. Sharing the
    # scale between qubit and qutrit keeps the two plots comparable.
    vmin = min(grid_qubit.min(), grid_qutrit.min())
    vmax = max(grid_qubit.max(), grid_qutrit.max())

    for grid, label, fname in [
        (grid_qubit, f"Qubit (d=2, m={QUBIT_M_MATCHED}, N=8)", "fig6_qubit_heatmap.png"),
        (grid_qutrit, f"Qutrit (d=3, m={QUTRIT_M_MATCHED}, N=9)", "fig7_qutrit_heatmap.png"),
    ]:
        plt.figure(figsize=(6, 5))
        im = plt.imshow(
            grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis",
            extent=[
                np.log10(HEATMAP_T1_US[0]), np.log10(HEATMAP_T1_US[-1]),
                np.log10(HEATMAP_TPHI_US[0]), np.log10(HEATMAP_TPHI_US[-1]),
            ],
        )
        plt.colorbar(im, label="Success probability $P_s$")
        plt.xlabel(r"$\log_{10}(T_1 / \mu s)$")
        plt.ylabel(r"$\log_{10}(T_\phi / \mu s)$")
        plt.title(f"$P_s$ heatmap: {label}")
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, fname)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  wrote {path}")


def fig8_purity() -> None:
    _, _, hist_qubit = run_grover_noisy(
        QUBIT_M_MATCHED, 2, MARKED_INDEX, DEFAULT_T1_US, DEFAULT_TPHI_US, DEFAULT_TG_US,
        return_history=True,
    )
    _, _, hist_qutrit = run_grover_noisy(
        QUTRIT_M_MATCHED, 3, MARKED_INDEX, DEFAULT_T1_US, DEFAULT_TPHI_US, DEFAULT_TG_US,
        return_history=True,
    )
    purity_qubit = [purity_of(r) for r in hist_qubit]
    purity_qutrit = [purity_of(r) for r in hist_qutrit]

    plt.figure(figsize=(6, 4))
    plt.plot(purity_qubit, "o-", label="Qubit (N=8)")
    plt.plot(purity_qutrit, "s-", label="Qutrit (N=9)")
    plt.xlabel("Grover iteration")
    plt.ylabel(r"Purity $\mathrm{Tr}(\rho^2)$")
    plt.title("State purity vs iteration (noisy circuit)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig8_purity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  wrote {path}")


def fig9_fidelity() -> None:
    _, _, hist_qubit_ideal = run_grover(QUBIT_M_MATCHED, 2, MARKED_INDEX, return_history=True)
    _, _, hist_qutrit_ideal = run_grover(QUTRIT_M_MATCHED, 3, MARKED_INDEX, return_history=True)
    _, _, hist_qubit_noisy = run_grover_noisy(
        QUBIT_M_MATCHED, 2, MARKED_INDEX, DEFAULT_T1_US, DEFAULT_TPHI_US, DEFAULT_TG_US,
        return_history=True,
    )
    _, _, hist_qutrit_noisy = run_grover_noisy(
        QUTRIT_M_MATCHED, 3, MARKED_INDEX, DEFAULT_T1_US, DEFAULT_TPHI_US, DEFAULT_TG_US,
        return_history=True,
    )
    fid_qubit = [fidelity_to_pure(rho, psi) for rho, psi in zip(hist_qubit_noisy, hist_qubit_ideal)]
    fid_qutrit = [fidelity_to_pure(rho, psi) for rho, psi in zip(hist_qutrit_noisy, hist_qutrit_ideal)]

    plt.figure(figsize=(6, 4))
    plt.plot(fid_qubit, "o-", label="Qubit (N=8)")
    plt.plot(fid_qutrit, "s-", label="Qutrit (N=9)")
    plt.xlabel("Grover iteration")
    plt.ylabel(r"Fidelity $\langle\psi_{\rm ideal}|\rho|\psi_{\rm ideal}\rangle$")
    plt.title("Fidelity to ideal trajectory vs iteration (noisy circuit)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig9_fidelity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Markdown table + summary rendering
# ---------------------------------------------------------------------------

def render_table1_markdown(ideal_rows: List[Dict], noisy_rows: List[Dict]) -> str:
    noisy_by_key = {(r["system"], r["m"]): r for r in noisy_rows}
    lines = [
        "| System | d | m | N=d^m | t_opt | Ideal Ps | Noisy Ps | Fidelity |",
        "| ------ | --: | --: | ------: | ------------: | ----------: | ----------: | -------: |",
    ]
    for r in ideal_rows:
        nr = noisy_by_key.get((r["system"], r["m"]))
        noisy_ps = f"{nr['noisy_Ps']:.4f}" if nr else "--"
        fid = f"{nr['fidelity']:.4f}" if nr else "--"
        lines.append(
            f"| {r['system']} | {r['d']} | {r['m']} | {r['N']} | {r['t_opt']} | "
            f"{r['ideal_Ps']:.4f} | {noisy_ps} | {fid} |"
        )
    return "\n".join(lines)


def render_table2_markdown(rows: List[Dict]) -> str:
    lines = [
        "| T1 | Tphi | Ps (qubit, N=8) | Ps (qutrit, N=9) | qutrit advantage |",
        "| -----: | -------: | ----------: | ----------: | ---------------: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['T1_us']:.0f} \u03bcs | {r['Tphi_us']:.0f} \u03bcs | "
            f"{r['Ps_qubit']:.4f} | {r['Ps_qutrit']:.4f} | {r['qutrit_advantage']:+.4f} |"
        )
    return "\n".join(lines)


def write_summary(
    ideal_rows: List[Dict], noisy_rows: List[Dict], matched_rows: List[Dict]
) -> str:
    """
    Answers Q1/Q2/Q3 FROM THE COMPUTED DATA, not from an assumed
    conclusion. Reads back what was actually written to the CSVs.
    """
    qubit_ideal = next(r for r in ideal_rows if r["system"] == "Qubit" and r["m"] == QUBIT_M_MATCHED)
    qutrit_ideal = next(r for r in ideal_rows if r["system"] == "Qutrit" and r["m"] == QUTRIT_M_MATCHED)
    qubit_noisy = next(r for r in noisy_rows if r["system"] == "Qubit" and r["m"] == QUBIT_M_MATCHED)
    qutrit_noisy = next(r for r in noisy_rows if r["system"] == "Qutrit" and r["m"] == QUTRIT_M_MATCHED)

    # Q1: does amplification work at all (ideal Ps >> 1/N)?
    q1_qubit = qubit_ideal["ideal_Ps"] > 1.0 / qubit_ideal["N"]
    q1_qutrit = qutrit_ideal["ideal_Ps"] > 1.0 / qutrit_ideal["N"]

    # Q2: larger search space per physical carrier -- trivially true by
    # construction (3^m > 2^m), stated here for completeness with actual
    # numbers, not just the inequality.
    q2_ratio = qutrit_ideal["N"] / qubit_ideal["N"]

    # Q3: does the qutrit's advantage (if any, at these N) survive noise?
    ideal_gap = qutrit_ideal["ideal_Ps"] - qubit_ideal["ideal_Ps"]
    noisy_gap = qutrit_noisy["noisy_Ps"] - qubit_noisy["noisy_Ps"]

    crossover = None
    prev_gap = None
    prev_val = None
    for row in matched_rows:
        gap = row["qutrit_advantage"]
        if prev_gap is not None and (prev_gap > 0) != (gap > 0):
            crossover = (prev_val, row["T1_us"])
        prev_gap, prev_val = gap, row["T1_us"]

    lines = [
        "# Summary: what the data actually shows",
        "",
        "## Q1 -- Does Ps increase through Grover amplification?",
        f"- Qubit (N={qubit_ideal['N']}): ideal Ps = {qubit_ideal['ideal_Ps']:.4f} vs "
        f"classical baseline 1/N = {1.0/qubit_ideal['N']:.4f} "
        f"-> amplification {'confirmed' if q1_qubit else 'NOT confirmed'}.",
        f"- Qutrit (N={qutrit_ideal['N']}): ideal Ps = {qutrit_ideal['ideal_Ps']:.4f} vs "
        f"classical baseline 1/N = {1.0/qutrit_ideal['N']:.4f} "
        f"-> amplification {'confirmed' if q1_qutrit else 'NOT confirmed'}.",
        "",
        "## Q2 -- Does the qutrit provide a larger search space per physical carrier?",
        f"- At the matched register sizes used throughout (qubit m={QUBIT_M_MATCHED}, "
        f"qutrit m={QUTRIT_M_MATCHED}): N_qutrit / N_qubit = {q2_ratio:.3f} "
        f"({qutrit_ideal['N']} vs {qubit_ideal['N']}) using ONE FEWER physical qudit.",
        "",
        "## Q3 -- Does the larger Hilbert space remain advantageous after decoherence?",
        f"- Ideal-case gap (qutrit - qubit): {ideal_gap:+.4f}",
        f"- Noisy-case gap at T1={DEFAULT_T1_US}us, Tphi={DEFAULT_TPHI_US}us: {noisy_gap:+.4f}",
    ]
    if crossover:
        lines.append(
            f"- Crossover detected in the matched T1=Tphi sweep between "
            f"{crossover[0]:.0f}us and {crossover[1]:.0f}us: the sign of the qutrit "
            f"advantage flips in this range (see results/matched_T1_Tphi_sweep.csv "
            f"and Table 2)."
        )
    else:
        signs = {row["qutrit_advantage"] > 0 for row in matched_rows}
        if len(signs) == 1:
            verdict = "qutrit ahead" if True in signs else "qubit ahead"
            lines.append(
                f"- No sign change detected across the matched T1=Tphi sweep "
                f"({MATCHED_DIAGONAL_US[0]:.0f}-{MATCHED_DIAGONAL_US[-1]:.0f}us): "
                f"{verdict} at every point tested in this range."
            )
        else:
            lines.append("- Sign of the advantage is mixed/non-monotonic across the sweep; inspect "
                          "results/matched_T1_Tphi_sweep.csv directly.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_all_benchmarks() -> None:
    _ensure_dirs()
    verify_matched_case()

    print("Computing Table 1 (ideal + noisy)...")
    ideal_rows = compute_ideal_table()
    noisy_rows = compute_noisy_table()
    _write_csv(ideal_rows, os.path.join(RESULTS_DIR, "ideal_grover.csv"))
    _write_csv(noisy_rows, os.path.join(RESULTS_DIR, "noisy_grover.csv"))
    with open(os.path.join(RESULTS_DIR, "summary_table1.md"), "w") as f:
        f.write(render_table1_markdown(ideal_rows, noisy_rows))
    print("  wrote results/summary_table1.md")

    print("Sweeping T1...")
    rows_T1 = sweep_T1()
    _write_csv(rows_T1, os.path.join(RESULTS_DIR, "T1_sweep.csv"))

    print("Sweeping Tphi...")
    rows_Tphi = sweep_Tphi()
    _write_csv(rows_Tphi, os.path.join(RESULTS_DIR, "Tphi_sweep.csv"))

    print("Sweeping matched T1=Tphi diagonal (Table 2)...")
    matched_rows = sweep_matched_diagonal()
    _write_csv(matched_rows, os.path.join(RESULTS_DIR, "matched_T1_Tphi_sweep.csv"))
    with open(os.path.join(RESULTS_DIR, "summary_table2.md"), "w") as f:
        f.write(render_table2_markdown(matched_rows))
    print("  wrote results/summary_table2.md")

    print("Running mesolve validation...")
    validation_rows = compute_validation_table()
    _write_csv(validation_rows, os.path.join(RESULTS_DIR, "validation.csv"))
    max_dist = max(r["trace_distance"] for r in validation_rows)
    print(f"  max trace distance vs mesolve across all validation points: {max_dist:.2e}")

    print("Generating figures...")
    fig1_ideal_grover()
    fig2_scaling()
    fig3_noise()
    fig4_T1(rows_T1)
    fig5_Tphi(rows_Tphi)
    fig6_and_fig7_heatmaps()
    fig8_purity()
    fig9_fidelity()

    print("Writing summary...")
    summary = write_summary(ideal_rows, noisy_rows, matched_rows)
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    run_all_benchmarks()
