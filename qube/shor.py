"""
Shor's factoring algorithm implementation.

Shor's algorithm factors integers in polynomial time on a quantum computer,
which would break RSA encryption. The algorithm uses quantum phase estimation
to find the period of modular exponentiation.

This module includes:
- Controlled modular multiplication circuits for N=15
- The complete Shor's algorithm for factoring 15
- Utilities for period extraction
"""

import numpy as np
from typing import List, Tuple, Optional

from .core import (
    applyGate, pushQubit, measureQubit, reset,
    get_workspace, set_workspace, get_namestack
)
from .gates import H_gate, X_gate, TOFF_gate
from .qft import QFT_inverse
from .utils import gcd, extract_period


# =============================================================================
# Controlled modular multiplication for N=15
# =============================================================================

def controlled_swap_basis_states(control: str, work_qubits: List[str], a: int, b: int):
    """
    Swap basis states |a⟩ and |b⟩ when control is |1⟩.

    Implements a transposition (a, b) in the permutation.
    Used to build controlled modular multiplication.

    Note: a and b should differ in exactly 2 bits for this implementation.

    Args:
        control: Control qubit name
        work_qubits: List of 4 work qubit names [W0, W1, W2, W3]
        a, b: Basis states to swap (integers 0-15)
    """
    W0, W1, W2, W3 = work_qubits
    W = [W0, W1, W2, W3]

    # Find differing bits
    diff = a ^ b
    diff_bits = [i for i in range(4) if (diff >> i) & 1]

    # Find matching bits and their required values (from 'a')
    match_bits = [i for i in range(4) if not ((diff >> i) & 1)]
    match_vals = [(a >> i) & 1 for i in match_bits]

    # Apply X gates to convert 0-controls to 1-controls
    for i, v in zip(match_bits, match_vals):
        if v == 0:
            applyGate(X_gate, W[i])

    # Use ancilla for 3-controlled operation
    m0, m1 = match_bits[0], match_bits[1]
    d0, d1 = diff_bits[0], diff_bits[1]

    pushQubit("_anc", [1, 0])
    applyGate(TOFF_gate, control, W[m0], "_anc")
    applyGate(TOFF_gate, "_anc", W[m1], W[d0])
    applyGate(TOFF_gate, "_anc", W[m1], W[d1])
    applyGate(TOFF_gate, control, W[m0], "_anc")
    measureQubit("_anc")

    # Undo X gates
    for i, v in zip(match_bits, match_vals):
        if v == 0:
            applyGate(X_gate, W[i])


def controlled_multiply_7_mod15(control: str, work_qubits: List[str], verbose: bool = False):
    """
    Controlled multiplication by 7 mod 15.

    Implements the permutation on the multiplicative group mod 15:
    Cycle 1: 1 → 7 → 4 → 13 → 1
    Cycle 2: 2 → 14 → 8 → 11 → 2

    Note: This implementation is correct only for inputs in the multiplicative
    group {1, 2, 4, 7, 8, 11, 13, 14}. Starting with |1⟩ ensures we stay in
    this group.

    Args:
        control: Control qubit name
        work_qubits: List of 4 work qubit names [W0, W1, W2, W3] (LSB to MSB)
        verbose: If True, print progress
    """
    if verbose:
        print(f"Applying controlled ×7 mod 15, control={control}")

    # Cycle 1: 1 → 7 → 4 → 13 → 1
    controlled_swap_basis_states(control, work_qubits, 1, 7)
    controlled_swap_basis_states(control, work_qubits, 1, 4)
    controlled_swap_basis_states(control, work_qubits, 1, 13)

    # Cycle 2: 2 → 14 → 8 → 11 → 2
    controlled_swap_basis_states(control, work_qubits, 2, 14)
    controlled_swap_basis_states(control, work_qubits, 2, 8)
    controlled_swap_basis_states(control, work_qubits, 2, 11)


def controlled_multiply_4_mod15(control: str, work_qubits: List[str], verbose: bool = False):
    """
    Controlled multiplication by 4 mod 15.

    Multiplication by 4 = 2² is a 2-bit left rotation mod 15:
    (w3, w2, w1, w0) → (w1, w0, w3, w2)

    Args:
        control: Control qubit name
        work_qubits: List of 4 work qubit names [W0, W1, W2, W3] (LSB to MSB)
        verbose: If True, print progress
    """
    W0, W1, W2, W3 = work_qubits

    if verbose:
        print(f"Applying controlled ×4 mod 15, control={control}")

    # Controlled SWAP W0 ↔ W2
    applyGate(TOFF_gate, control, W0, W2)
    applyGate(TOFF_gate, control, W2, W0)
    applyGate(TOFF_gate, control, W0, W2)

    # Controlled SWAP W1 ↔ W3
    applyGate(TOFF_gate, control, W1, W3)
    applyGate(TOFF_gate, control, W3, W1)
    applyGate(TOFF_gate, control, W1, W3)


# =============================================================================
# Shor's algorithm
# =============================================================================

def shor_factor_15(verbose: bool = False) -> Optional[Tuple[int, int]]:
    """
    Shor's algorithm to factor N=15 using a=7.

    The algorithm:
    1. Initialize work register to |1⟩
    2. Put control register in superposition
    3. Apply controlled modular exponentiations (7^1, 7^2, 7^4, 7^8)
    4. Apply inverse QFT to control register
    5. Measure and extract period using continued fractions
    6. Compute factors from the period

    Args:
        verbose: If True, print detailed progress

    Returns:
        Tuple (factor1, factor2) if successful, or None if failed
    """
    reset()

    N = 15
    a = 7
    n_control = 4  # Control register size (determines precision)

    if verbose:
        print(f"Shor's Algorithm: Factoring N={N} with a={a}")
        print(f"Using {n_control} control qubits and 4 work qubits")
        print()

    # Initialize work register to |1⟩ = |0001⟩
    # W0 (LSB) = 1, W1 = W2 = W3 = 0
    pushQubit("W0", [0, 1])  # LSB = 1
    pushQubit("W1", [1, 0])
    pushQubit("W2", [1, 0])
    pushQubit("W3", [1, 0])  # MSB

    if verbose:
        print("Work register initialized to |1⟩")

    # Initialize control register and apply Hadamards
    # Note: C0 is LSB (controls 7^1), C3 is MSB (controls 7^8)
    control_qubits = []
    for i in range(n_control):
        name = f"C{i}"
        pushQubit(name, [1, 0])
        applyGate(H_gate, name)
        control_qubits.append(name)

    if verbose:
        print(f"Control register in superposition: {control_qubits}")
        print()

    # Apply controlled modular exponentiations
    # C0 controls ×7^1 = ×7
    # C1 controls ×7^2 = ×4
    # C2 controls ×7^4 = ×1 (identity, skip)
    # C3 controls ×7^8 = ×1 (identity, skip)

    work_qubits = ["W0", "W1", "W2", "W3"]

    if verbose:
        print("Applying controlled modular exponentiations:")
        print("  C0: controlled ×7 mod 15")
    controlled_multiply_7_mod15("C0", work_qubits)

    if verbose:
        print("  C1: controlled ×4 mod 15 (= ×7² mod 15)")
    controlled_multiply_4_mod15("C1", work_qubits)

    if verbose:
        print("  C2: controlled ×1 (identity, skipped)")
        print("  C3: controlled ×1 (identity, skipped)")
        print()

    # Apply inverse QFT to control register
    # IMPORTANT: QFT expects [MSB, ..., LSB] order, but control_qubits is [C0, C1, C2, C3]
    # where C0 is LSB. So we must reverse to get [C3, C2, C1, C0] = [MSB, ..., LSB]
    if verbose:
        print("Applying inverse QFT to control register...")
    QFT_inverse(control_qubits[::-1])

    if verbose:
        print()
        print("Measuring control register:")

    # Measure control register
    measurement = 0
    for i in range(n_control - 1, -1, -1):
        bit = int(measureQubit(control_qubits[i]))
        measurement = measurement * 2 + bit
        if verbose:
            print(f"  {control_qubits[i]} = {bit}")

    if verbose:
        print(f"\nMeasurement result: {measurement} (decimal)")
        print(f"This represents phase ≈ {measurement}/{2 ** n_control} = {measurement / 2 ** n_control}")

    # Clean up work register
    for q in reversed(work_qubits):
        measureQubit(q)

    # Extract period using continued fractions
    r = extract_period(measurement, n_control, N, a)

    if verbose:
        print(f"\nExtracted period candidate: r = {r}")

    if r is None or r % 2 == 1:
        if verbose:
            print("Period extraction failed or r is odd. Algorithm failed this run.")
        return None

    # Compute factors
    x = pow(a, r // 2, N)
    factor1 = gcd(x - 1, N)
    factor2 = gcd(x + 1, N)

    if verbose:
        print(f"\nComputing factors:")
        print(f"  a^(r/2) mod N = {a}^{r // 2} mod {N} = {x}")
        print(f"  gcd({x} - 1, {N}) = gcd({x - 1}, {N}) = {factor1}")
        print(f"  gcd({x} + 1, {N}) = gcd({x + 1}, {N}) = {factor2}")

    # Check if we found non-trivial factors
    if factor1 != 1 and factor1 != N:
        if verbose:
            print(f"\n✓ Success! Found factors: {factor1} × {N // factor1} = {N}")
        return (factor1, N // factor1)
    elif factor2 != 1 and factor2 != N:
        if verbose:
            print(f"\n✓ Success! Found factors: {factor2} × {N // factor2} = {N}")
        return (factor2, N // factor2)
    else:
        if verbose:
            print("\n✗ Found only trivial factors. Algorithm failed this run.")
        return None


def shor_factor_15_multiple_runs(
    num_runs: int = 10,
    verbose: bool = False
) -> Optional[Tuple[int, int]]:
    """
    Run Shor's algorithm multiple times until factors are found.

    Since Shor's algorithm is probabilistic, multiple runs may be needed.

    Args:
        num_runs: Maximum number of attempts
        verbose: If True, print progress

    Returns:
        Tuple (factor1, factor2) if successful, or None if all runs failed
    """
    for run in range(num_runs):
        if verbose:
            print(f"\n{'=' * 40}")
            print(f"Run {run + 1}/{num_runs}")
            print('=' * 40)

        result = shor_factor_15(verbose=verbose)

        if result is not None:
            if verbose:
                print(f"\nSuccess on run {run + 1}!")
            return result

    if verbose:
        print(f"\nFailed to find factors in {num_runs} runs")
    return None


# =============================================================================
# Classical helper for verification
# =============================================================================

def classical_mod_mult(x: int, a: int, N: int) -> int:
    """Classical modular multiplication for verification."""
    return (x * a) % N


def verify_modular_multiplication():
    """
    Verify the controlled multiplication circuits against classical computation.

    Returns:
        True if all tests pass
    """
    from .core import reset, get_workspace

    print("Verifying controlled ×7 mod 15:")
    test_values = [1, 2, 4, 7, 8, 11, 13, 14]
    all_pass = True

    for x in test_values:
        reset()

        # Encode x in work register (binary)
        pushQubit("W0", [1 - (x & 1), x & 1])
        pushQubit("W1", [1 - ((x >> 1) & 1), (x >> 1) & 1])
        pushQubit("W2", [1 - ((x >> 2) & 1), (x >> 2) & 1])
        pushQubit("W3", [1 - ((x >> 3) & 1), (x >> 3) & 1])

        # Control qubit = |1⟩
        pushQubit("C", [0, 1])

        controlled_multiply_7_mod15("C", ["W0", "W1", "W2", "W3"])

        # Measure result
        result = 0
        for i, q in enumerate(["W0", "W1", "W2", "W3"]):
            bit = int(measureQubit(q))
            result += bit * (2 ** i)
        measureQubit("C")

        expected = classical_mod_mult(x, 7, 15)
        status = "✓" if result == expected else "✗"
        print(f"  {x} × 7 mod 15: got {result}, expected {expected} {status}")
        if result != expected:
            all_pass = False

    print("\nVerifying controlled ×4 mod 15:")
    for x in test_values:
        reset()

        pushQubit("W0", [1 - (x & 1), x & 1])
        pushQubit("W1", [1 - ((x >> 1) & 1), (x >> 1) & 1])
        pushQubit("W2", [1 - ((x >> 2) & 1), (x >> 2) & 1])
        pushQubit("W3", [1 - ((x >> 3) & 1), (x >> 3) & 1])
        pushQubit("C", [0, 1])

        controlled_multiply_4_mod15("C", ["W0", "W1", "W2", "W3"])

        result = 0
        for i, q in enumerate(["W0", "W1", "W2", "W3"]):
            bit = int(measureQubit(q))
            result += bit * (2 ** i)
        measureQubit("C")

        expected = classical_mod_mult(x, 4, 15)
        status = "✓" if result == expected else "✗"
        print(f"  {x} × 4 mod 15: got {result}, expected {expected} {status}")
        if result != expected:
            all_pass = False

    return all_pass
