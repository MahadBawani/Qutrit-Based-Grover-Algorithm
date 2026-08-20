"""
qudit_grover
============
A QuTiP-based toolkit for simulating Grover's search algorithm on general
qudits (dimension d >= 2), using full density-matrix formalism with Kraus
channel noise models.

Everything is parameterized by the qudit dimension `d`, so d=2 recovers
standard qubit Grover and d=3 gives the qutrit case this project targets,
with no code duplication between the two -- the noisy/ideal, qutrit-vs-qubit
benchmark just calls the same functions with a different `d`.

Submodules
----------
states.py       Basis states, superpositions, mixed states, generalized
                Bell/GHZ states.                               (implemented)
gates.py        Chrestenson gate, generalized X/Z (clock-and-shift) gates,
                CSUM/CZ/SWAP, and embed() for placing local gates onto an
                m-qudit register.                              (implemented)
noise.py        Physically-anchored transmon Kraus channels (bosonic
                cascade amplitude damping + pure dephasing) for a single
                qudit, parameterized by T1/Tphi.                (implemented)
grover.py       Ideal (noiseless) oracle, diffusion operator, and
                run_grover() for a pure-state m-qudit register. (implemented)
grover_noise.py Noisy Grover: same oracle/diffusion, density-matrix
                propagation with noise.py's channels applied to every
                register qudit after each gate layer.          (implemented)
simulation.py   Hardcoded ideal-vs-noisy, qudit-vs-qubit benchmark run:
                writes results/*.csv and figures/*.png.         (implemented)

Memory notes
------------
States are QuTiP Qobj's stored in QuTiP's default sparse (CSR) backend;
avoid calling `.full()` on large multi-qudit objects unless you specifically
need a dense NumPy array. `states.clear_state_cache()` releases the small
internal cache of single-qudit basis kets, and `gates.clear_gate_cache()`
releases the cached base gate matrices -- call both together between sweeps
over different dimensions `d` to keep memory flat during long benchmark runs.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .states import (
    basis_state,
    comp_basis,
    density_from_ket,
    ghz_state,
    is_valid_state,
    maximally_mixed,
    purity,
    qudit_bell_state,
    random_density_matrix,
    random_pure_state,
    uniform_superposition,
    clear_state_cache,
)

from .gates import (
    omega,
    shift_gate,
    clock_gate,
    chrestenson_gate,
    phase_gate,
    csum_gate,
    cz_gate,
    swap_gate,
    embed,
    chrestenson_all,
    identity,
    clear_gate_cache,
)

from .noise import (
    decay_probability,
    dephasing_probability,
    Tphi_from_T1_T2,
    transmon_T1_ladder,
    transmon_Tphi_ladder,
    amplitude_damping_kraus,
    dephasing_kraus,
    compose_channels,
    check_completeness,
    apply_channel,
    transmon_qudit_channel,
    sample_effective_T1,
    validate_amplitude_damping_vs_mesolve,
)

from .grover import (
    oracle,
    diffusion_operator,
    optimal_iterations,
    run_grover,
    success_probability,
    success_probability_trace,
)

from .grover_noise import (
    embed_channel,
    apply_register_noise,
    run_grover_noisy,
    success_probability_mixed,
    success_probability_trace_mixed,
)

from .simulation import run_all_benchmarks

__all__ = [
    # states.py
    "basis_state",
    "comp_basis",
    "density_from_ket",
    "ghz_state",
    "is_valid_state",
    "maximally_mixed",
    "purity",
    "qudit_bell_state",
    "random_density_matrix",
    "random_pure_state",
    "uniform_superposition",
    "clear_state_cache",
    # gates.py
    "omega",
    "shift_gate",
    "clock_gate",
    "chrestenson_gate",
    "phase_gate",
    "csum_gate",
    "cz_gate",
    "swap_gate",
    "embed",
    "chrestenson_all",
    "identity",
    "clear_gate_cache",
    # noise.py
    "decay_probability",
    "dephasing_probability",
    "Tphi_from_T1_T2",
    "transmon_T1_ladder",
    "transmon_Tphi_ladder",
    "amplitude_damping_kraus",
    "dephasing_kraus",
    "compose_channels",
    "check_completeness",
    "apply_channel",
    "transmon_qudit_channel",
    "sample_effective_T1",
    "validate_amplitude_damping_vs_mesolve",
    # grover.py
    "oracle",
    "diffusion_operator",
    "optimal_iterations",
    "run_grover",
    "success_probability",
    "success_probability_trace",
    # grover_noise.py
    "embed_channel",
    "apply_register_noise",
    "run_grover_noisy",
    "success_probability_mixed",
    "success_probability_trace_mixed",
    # simulation.py
    "run_all_benchmarks",
]

# benchmark.py doesn't exist yet -- uncomment once implemented.
# from .benchmark import *
