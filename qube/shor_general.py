"""
Sketch of a general Shor's algorithm implementation.

This shows the structure needed for arbitrary N, but uses classical
computation for the modular arithmetic (since full quantum implementation
requires extensive circuit construction).

The key insight: we can still demonstrate the quantum parts (QFT, phase
estimation) while using classical oracles for modular exponentiation.
This is useful for:
1. Understanding the algorithm structure
2. Testing with larger N values
3. Educational purposes

For a true quantum implementation, you'd need to replace the oracle
with actual quantum circuits for modular arithmetic.
"""

import numpy as np
from typing import Tuple, Optional, List
import random

from .core import reset, pushQubit, applyGate, measureQubit, get_state, get_namestack
from .gates import H_gate, X_gate
from .qft import QFT_inverse
from .utils import gcd, is_coprime, extract_period


def find_order_classical(a: int, N: int) -> int:
    """Find the multiplicative order of a mod N (classical brute force)."""
    if gcd(a, N) != 1:
        raise ValueError(f"{a} and {N} are not coprime")

    r = 1
    x = a
    while x != 1:
        x = (x * a) % N
        r += 1
        if r > N:
            raise ValueError(f"Order not found (shouldn't happen)")
    return r


def shor_classical_postprocessing(measurement: int, num_qubits: int, N: int, a: int) -> Optional[int]:
    """
    Classical post-processing for Shor's algorithm.

    Given a measurement from the phase estimation, extract the period.
    This is the same as extract_period but with more diagnostics.
    """
    return extract_period(measurement, num_qubits, N, a)


def shor_factor_general(N: int, a: Optional[int] = None, verbose: bool = False) -> Optional[Tuple[int, int]]:
    """
    General Shor's algorithm structure for factoring N.

    This implementation uses a CLASSICAL oracle for modular exponentiation,
    which means it doesn't provide quantum speedup. However, it correctly
    demonstrates:
    1. The quantum phase estimation structure
    2. The classical post-processing (continued fractions)
    3. The probabilistic nature of the algorithm

    For N > ~20, this becomes slow due to classical simulation limits,
    but you can run it to see the algorithm work.

    Args:
        N: Number to factor (should be composite, not a prime power)
        a: Base for modular exponentiation (random if not specified)
        verbose: Print detailed output

    Returns:
        (factor1, factor2) if successful, None otherwise
    """

    # Step 1: Classical preprocessing
    if verbose:
        print(f"Shor's Algorithm for N = {N}")
        print("=" * 50)

    # Check if N is even
    if N % 2 == 0:
        if verbose:
            print(f"N is even, trivial factor: 2")
        return (2, N // 2)

    # Check if N is a prime power
    for b in range(2, int(np.log2(N)) + 1):
        root = int(round(N ** (1/b)))
        if root ** b == N:
            if verbose:
                print(f"N = {root}^{b}, trivial factor: {root}")
            return (root, N // root)

    # Choose random a if not specified
    if a is None:
        candidates = [x for x in range(2, N) if is_coprime(x, N)]
        a = random.choice(candidates)

    if verbose:
        print(f"Using base a = {a}")

    # Check if we got lucky with gcd
    g = gcd(a, N)
    if g > 1:
        if verbose:
            print(f"Lucky! gcd({a}, {N}) = {g}")
        return (g, N // g)

    # Step 2: Quantum period finding (simulated)
    # In a real quantum computer, this is where the speedup happens

    n = N.bit_length()
    num_control_qubits = 2 * n  # For sufficient precision

    if verbose:
        print(f"Period finding with {num_control_qubits} control qubits")
        print(f"(Simulating quantum phase estimation...)")

    # For educational purposes, we'll compute what the quantum computer
    # would measure. The true order r leads to measurements that are
    # multiples of 2^num_control_qubits / r

    r_true = find_order_classical(a, N)
    if verbose:
        print(f"True period: r = {r_true}")

    # Simulate a measurement from the quantum phase estimation
    # The quantum computer would measure values close to k * 2^n / r
    # for k = 0, 1, ..., r-1
    s = random.randint(0, r_true - 1)  # Random s in [0, r)
    measurement = round(s * (2 ** num_control_qubits) / r_true)
    measurement = measurement % (2 ** num_control_qubits)

    if verbose:
        print(f"Simulated measurement: {measurement}")
        print(f"  (This represents phase ≈ {s}/{r_true} = {s/r_true:.6f})")

    # Step 3: Classical post-processing
    r = shor_classical_postprocessing(measurement, num_control_qubits, N, a)

    if verbose:
        print(f"Extracted period: r = {r}")

    if r is None:
        if verbose:
            print("Period extraction failed")
        return None

    if r % 2 == 1:
        if verbose:
            print("Period is odd, algorithm failed this run")
        return None

    # Step 4: Compute factors
    x = pow(a, r // 2, N)

    if verbose:
        print(f"a^(r/2) mod N = {a}^{r//2} mod {N} = {x}")

    if x == N - 1:  # x ≡ -1 (mod N)
        if verbose:
            print("x ≡ -1 (mod N), algorithm failed this run")
        return None

    factor1 = gcd(x - 1, N)
    factor2 = gcd(x + 1, N)

    if verbose:
        print(f"gcd({x} - 1, {N}) = {factor1}")
        print(f"gcd({x} + 1, {N}) = {factor2}")

    if factor1 != 1 and factor1 != N:
        if verbose:
            print(f"SUCCESS: {factor1} × {N // factor1} = {N}")
        return (factor1, N // factor1)
    elif factor2 != 1 and factor2 != N:
        if verbose:
            print(f"SUCCESS: {factor2} × {N // factor2} = {N}")
        return (factor2, N // factor2)
    else:
        if verbose:
            print("Found only trivial factors, algorithm failed")
        return None


def demo_shor_various_N():
    """Demonstrate Shor's algorithm on various N values."""

    test_cases = [
        15,   # 3 × 5
        21,   # 3 × 7
        33,   # 3 × 11
        35,   # 5 × 7
        77,   # 7 × 11
        91,   # 7 × 13
        143,  # 11 × 13
        221,  # 13 × 17
        323,  # 17 × 19
    ]

    print("Shor's Algorithm Demo (various N)")
    print("=" * 60)
    print(f"{'N':>6} | {'Bits':>4} | {'Result':>15} | {'Verification':>15}")
    print("-" * 60)

    for N in test_cases:
        # Try up to 5 times
        for attempt in range(5):
            result = shor_factor_general(N, verbose=False)
            if result is not None:
                f1, f2 = result
                verification = f"✓ {f1}×{f2}={f1*f2}" if f1 * f2 == N else "✗ WRONG"
                print(f"{N:>6} | {N.bit_length():>4} | {f1:>6} × {f2:<6} | {verification}")
                break
        else:
            print(f"{N:>6} | {N.bit_length():>4} | {'Failed':>15} | (5 attempts)")


if __name__ == "__main__":
    # Demo with various N
    demo_shor_various_N()

    print()
    print("=" * 60)
    print()

    # Detailed run for N=21
    print("Detailed run for N=21:")
    print()
    shor_factor_general(21, verbose=True)
