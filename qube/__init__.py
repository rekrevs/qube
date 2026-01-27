"""
Qube - A quantum computing simulator in Python.

This package provides a simple state-vector simulator for quantum circuits,
along with implementations of fundamental quantum algorithms.

Modules:
    gates   - Quantum gate definitions (H, X, Y, Z, CNOT, SWAP, etc.)
    core    - Core simulation functionality (pushQubit, applyGate, measureQubit)
    qft     - Quantum Fourier Transform
    grover  - Grover's search algorithm
    shor    - Shor's factoring algorithm
    utils   - Utility functions (continued fractions, state comparison)

Quick Start:
    >>> from qube import *
    >>> reset()
    >>> pushQubit("q", [1, 0])
    >>> applyGate(H_gate, "q")
    >>> print(measureQubit("q"))  # "0" or "1" with 50% probability each
"""

# Core functionality
from .core import (
    # State management
    reset,
    get_state,
    get_workspace,
    set_workspace,
    get_namestack,
    # Qubit operations
    pushQubit,
    tosQubit,
    applyGate,
    probQubit,
    measureQubit,
    peekQubit,
    # Composite gates
    toffoli_decomposed,
    CP_decomposed,
    TOFF3,
    TOFFn,
)

# Gates
from .gates import (
    # Single-qubit gates
    X_gate,
    Y_gate,
    Z_gate,
    H_gate,
    S_gate,
    T_gate,
    Tinv_gate,
    I_gate,
    P_gate,
    Rx_gate,
    Ry_gate,
    Rz_gate,
    # Two-qubit gates
    CNOT_gate,
    CZ_gate,
    SWAP_gate,
    CP_gate,
    # Three-qubit gates
    TOFF_gate,
)

# QFT
from .qft import (
    QFT,
    QFT_inverse,
    QFT_decomposed,
    QFT_inverse_decomposed,
)

# Utilities
from .utils import (
    allclose_up_to_global_phase,
    state_fidelity,
    gcd,
    is_coprime,
    mod_inverse,
    continued_fraction_expansion,
    convergents,
    extract_period,
    int_to_bits,
    bits_to_int,
)

# Algorithms
from .grover import (
    grover_search,
    grover_search_for_value,
    create_single_target_oracle,
    diffusion_operator,
    zero_phase_oracle,
)

from .shor import (
    shor_factor_15,
    shor_factor_15_multiple_runs,
    controlled_multiply_7_mod15,
    controlled_multiply_4_mod15,
    verify_modular_multiplication,
)

# Draper QFT Adder
from .draper import (
    draper_add,
    draper_add_constant,
    draper_subtract,
    draper_subtract_constant,
    phi_add_constant,
    phi_add_register,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "reset",
    "get_state",
    "get_workspace",
    "set_workspace",
    "get_namestack",
    "pushQubit",
    "tosQubit",
    "applyGate",
    "probQubit",
    "measureQubit",
    "peekQubit",
    "toffoli_decomposed",
    "CP_decomposed",
    "TOFF3",
    "TOFFn",
    # Gates
    "X_gate",
    "Y_gate",
    "Z_gate",
    "H_gate",
    "S_gate",
    "T_gate",
    "Tinv_gate",
    "I_gate",
    "P_gate",
    "Rx_gate",
    "Ry_gate",
    "Rz_gate",
    "CNOT_gate",
    "CZ_gate",
    "SWAP_gate",
    "CP_gate",
    "TOFF_gate",
    # QFT
    "QFT",
    "QFT_inverse",
    "QFT_decomposed",
    "QFT_inverse_decomposed",
    # Utils
    "allclose_up_to_global_phase",
    "state_fidelity",
    "gcd",
    "is_coprime",
    "mod_inverse",
    "continued_fraction_expansion",
    "convergents",
    "extract_period",
    "int_to_bits",
    "bits_to_int",
    # Grover
    "grover_search",
    "grover_search_for_value",
    "create_single_target_oracle",
    "diffusion_operator",
    "zero_phase_oracle",
    # Shor
    "shor_factor_15",
    "shor_factor_15_multiple_runs",
    "controlled_multiply_7_mod15",
    "controlled_multiply_4_mod15",
    "verify_modular_multiplication",
    # Draper Adder
    "draper_add",
    "draper_add_constant",
    "draper_subtract",
    "draper_subtract_constant",
    "phi_add_constant",
    "phi_add_register",
]
