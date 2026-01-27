"""
Analysis and building blocks for scaling Shor's algorithm.

This module explores what's needed to factor larger numbers:
1. General modular arithmetic circuits
2. Resource estimation
3. Classical simulation limits

Key insight: The quantum algorithm is polynomial, but classical
simulation is exponential. That's why quantum computers matter.
"""

import numpy as np
from typing import Tuple, List, Optional
import math


# =============================================================================
# Resource Estimation
# =============================================================================

def estimate_shor_resources(N: int) -> dict:
    """
    Estimate resources needed to factor N using Shor's algorithm.

    Args:
        N: Number to factor

    Returns:
        Dictionary with resource estimates
    """
    n = N.bit_length()  # Bits needed to represent N

    # Control register: 2n qubits for sufficient precision
    # (we need phase estimation accurate to O(1/N²))
    control_qubits = 2 * n

    # Work register: n qubits to hold values mod N
    work_qubits = n

    # Ancilla qubits for modular arithmetic (varies by implementation)
    # Beauregard's circuit: ~n ancillas
    # Standard reversible: ~2n ancillas
    ancilla_qubits_beauregard = n
    ancilla_qubits_standard = 2 * n

    # Gate counts (approximate)
    # QFT: O(n²) gates
    qft_gates = control_qubits * (control_qubits + 1) // 2

    # Modular exponentiation: 2n controlled modular multiplications
    # Each multiplication: O(n²) gates for addition-based, or O(n³) for naive
    # Using Beauregard's QFT-based addition: O(n²) per multiplication
    mod_exp_gates_beauregard = 2 * n * n * n  # O(n³)
    mod_exp_gates_naive = 2 * n * n * n * n   # O(n⁴)

    # Classical simulation cost
    total_qubits = control_qubits + work_qubits + ancilla_qubits_beauregard
    # Use log to avoid overflow for large qubit counts
    # memory = 2^total_qubits * 16 bytes
    # log2(memory_in_GB) = total_qubits + 4 - 30 = total_qubits - 26
    log2_memory_gb = total_qubits + 4 - 30  # 16 bytes = 2^4, 1GB = 2^30

    if total_qubits <= 60:
        state_vector_size = 2 ** total_qubits
        memory_bytes = state_vector_size * 16
        memory_gb = memory_bytes / (1024**3)
        memory_tb = memory_bytes / (1024**4)
    else:
        state_vector_size = float('inf')
        memory_gb = 2 ** log2_memory_gb if log2_memory_gb < 100 else float('inf')
        memory_tb = memory_gb / 1024

    return {
        "N": N,
        "bits": n,
        "control_qubits": control_qubits,
        "work_qubits": work_qubits,
        "ancilla_qubits_beauregard": ancilla_qubits_beauregard,
        "ancilla_qubits_standard": ancilla_qubits_standard,
        "total_qubits_beauregard": total_qubits,
        "total_qubits_standard": control_qubits + work_qubits + ancilla_qubits_standard,
        "qft_gates": qft_gates,
        "mod_exp_gates_beauregard": mod_exp_gates_beauregard,
        "mod_exp_gates_naive": mod_exp_gates_naive,
        "state_vector_size": state_vector_size,
        "memory_gb": memory_gb,
        "memory_tb": memory_tb,
        "log2_memory_gb": log2_memory_gb,
    }


def print_scaling_analysis():
    """Print scaling analysis for various problem sizes."""

    print("=" * 80)
    print("SHOR'S ALGORITHM SCALING ANALYSIS")
    print("=" * 80)
    print()

    # Test cases: from toy to RSA
    test_cases = [
        (15, "Toy example (current)"),
        (21, "Small composite"),
        (143, "11 × 13"),
        (1000003, "~20-bit number"),
        (2**32 - 5, "~32-bit number"),
        (2**64, "~64-bit (small RSA)"),
        (2**512, "512-bit (weak RSA)"),
        (2**1024, "1024-bit (deprecated RSA)"),
        (2**2048, "2048-bit (current RSA)"),
        (2**4096, "4096-bit (strong RSA)"),
    ]

    print(f"{'N':>20} | {'Bits':>5} | {'Qubits':>7} | {'Memory':>15} | Description")
    print("-" * 80)

    for N, desc in test_cases:
        r = estimate_shor_resources(N)

        if r["memory_tb"] > 1:
            mem_str = f"{r['memory_tb']:.0e} TB"
        elif r["memory_gb"] > 1:
            mem_str = f"{r['memory_gb']:.1f} GB"
        else:
            mem_str = f"{r['memory_gb']*1024:.1f} MB"

        # For very large numbers, the state vector is impossibly large
        if r["bits"] > 100:
            mem_str = "impossible"

        print(f"{N:>20} | {r['bits']:>5} | {r['total_qubits_beauregard']:>7} | {mem_str:>15} | {desc}")

    print()
    print("Key observations:")
    print("  • Classical simulation memory: O(2^n) - doubles with each additional bit")
    print("  • Quantum qubits needed: O(n) - linear in bits")
    print("  • ~50 qubits is the classical simulation limit (~16 PB memory)")
    print("  • RSA-2048 needs ~6000 qubits but cannot be classically simulated")
    print()


# =============================================================================
# Building Blocks for General Modular Arithmetic
# =============================================================================

def controlled_modular_add_classical(a: int, x: int, N: int) -> int:
    """Classical modular addition: (x + a) mod N."""
    return (x + a) % N


def controlled_modular_mult_classical(a: int, x: int, N: int) -> int:
    """Classical modular multiplication: (a * x) mod N."""
    return (a * x) % N


def modular_exp_classical(a: int, exponent: int, N: int) -> int:
    """Classical modular exponentiation: a^exponent mod N."""
    return pow(a, exponent, N)


def build_multiplication_permutation(a: int, N: int) -> List[int]:
    """
    Build the permutation table for multiplication by a mod N.

    For quantum implementation, we need to know where each basis
    state maps to under the transformation |x⟩ → |ax mod N⟩.

    This only works for x coprime to N (the multiplicative group).
    """
    from .utils import gcd

    permutation = []
    for x in range(N):
        if gcd(x, N) == 1:  # x is in multiplicative group
            permutation.append((x, (a * x) % N))
        else:
            permutation.append((x, x))  # Identity for non-coprime elements

    return permutation


def find_cycle_structure(a: int, N: int) -> List[List[int]]:
    """
    Find the cycle structure of multiplication by a mod N.

    This is useful for decomposing the permutation into transpositions
    (which is what we did for N=15).
    """
    from .utils import gcd

    visited = set()
    cycles = []

    for start in range(1, N):
        if start in visited or gcd(start, N) != 1:
            continue

        cycle = []
        x = start
        while x not in visited:
            visited.add(x)
            cycle.append(x)
            x = (a * x) % N

        if len(cycle) > 1:
            cycles.append(cycle)

    return cycles


# =============================================================================
# Reversible Classical Circuits (Building Blocks)
# =============================================================================

class ReversibleCircuit:
    """
    Represents a reversible classical circuit that can be turned into
    a quantum circuit.

    In a real implementation, these would generate actual quantum gates.
    Here we just track the classical transformations for understanding.
    """

    @staticmethod
    def controlled_add(ctrl: bool, a: int, x: int, n_bits: int) -> int:
        """
        Controlled addition: if ctrl, return x + a (mod 2^n_bits).

        Quantum implementation uses:
        - Ripple-carry adder: O(n) depth, O(n) gates
        - Carry-lookahead: O(log n) depth, O(n log n) gates
        - QFT-based (Draper): O(n) gates, uses rotations
        """
        if ctrl:
            return (x + a) % (2 ** n_bits)
        return x

    @staticmethod
    def controlled_mod_add(ctrl: bool, a: int, x: int, N: int, n_bits: int) -> int:
        """
        Controlled modular addition: if ctrl, return (x + a) mod N.

        Quantum implementation:
        1. Add a: x → x + a
        2. Subtract N: x + a → x + a - N
        3. If negative (MSB=1), add N back
        4. Uncompute the comparison
        """
        if ctrl:
            return (x + a) % N
        return x

    @staticmethod
    def controlled_mod_mult(ctrl: bool, a: int, x: int, N: int) -> int:
        """
        Controlled modular multiplication: if ctrl, return (a * x) mod N.

        Quantum implementation (requires x coprime to N):
        Uses the identity: a*x = Σ_i (2^i * a * x_i) where x = Σ x_i * 2^i

        1. For each bit i of x:
           - Controlled on x_i: add (2^i * a mod N) to accumulator
        2. Swap accumulator with x register
        3. Uncompute (multiply by a^(-1) mod N)
        """
        if ctrl:
            return (a * x) % N
        return x


# =============================================================================
# Demonstration: What changes for N=21
# =============================================================================

def analyze_n21():
    """
    Analyze what's needed to factor N=21 (= 3 × 7).

    This is the next step up from N=15.
    """
    N = 21

    # Find a suitable base
    from .utils import gcd
    candidates = [a for a in range(2, N) if gcd(a, N) == 1]
    print(f"Candidates for base a (coprime to {N}): {candidates}")

    # Find periods
    print(f"\nPeriod analysis for N={N}:")
    for a in candidates[:5]:  # First few
        # Find period by brute force
        for r in range(1, N):
            if pow(a, r, N) == 1:
                print(f"  a={a}: period r={r}, r is {'even' if r % 2 == 0 else 'odd'}")
                if r % 2 == 0:
                    x = pow(a, r // 2, N)
                    f1, f2 = gcd(x - 1, N), gcd(x + 1, N)
                    print(f"         a^(r/2) = {x}, gcd({x}-1, {N})={f1}, gcd({x}+1, {N})={f2}")
                break

    # Cycle structure for a=2
    a = 2
    cycles = find_cycle_structure(a, N)
    print(f"\nCycle structure for ×{a} mod {N}:")
    for cycle in cycles:
        print(f"  {' → '.join(map(str, cycle))} → {cycle[0]}")

    # Resources
    r = estimate_shor_resources(N)
    print(f"\nResources for N={N}:")
    print(f"  Qubits needed: {r['total_qubits_beauregard']}")
    print(f"  Classical simulation memory: {r['memory_gb']:.2f} GB")


# =============================================================================
# Approaches to Scaling
# =============================================================================

def describe_scaling_approaches():
    """Describe different approaches to implement larger Shor circuits."""

    approaches = """
APPROACHES TO SCALING SHOR'S ALGORITHM
======================================

1. HARDCODED PERMUTATIONS (Current approach for N=15)
   - Pro: Simple, educational
   - Con: Doesn't scale - circuit size exponential in bits
   - Use: Only for tiny examples

2. REVERSIBLE ADDER CIRCUITS
   - Build modular multiplication from modular addition
   - Build modular addition from binary addition + comparison
   - Ripple-carry: O(n) depth, simple
   - Carry-lookahead: O(log n) depth, more complex

3. QFT-BASED ARITHMETIC (Beauregard/Draper)
   - Use QFT to do addition in the "phase basis"
   - Add by applying controlled rotations
   - More gates but shorter depth
   - Can do addition with no ancillas!

4. WINDOWED ARITHMETIC
   - Pre-compute lookup tables for small windows
   - Trade space for time
   - Good for specific N values

5. APPROXIMATE/NOISY METHODS
   - For near-term quantum computers
   - Variational approaches
   - Error mitigation

CLASSICAL SIMULATION OPTIMIZATIONS
==================================

1. SPARSE STATE VECTORS
   - Most amplitudes are zero after each step
   - Only store non-zero amplitudes
   - Works well for structured circuits

2. TENSOR NETWORKS
   - Represent state as network of small tensors
   - MPS (Matrix Product States) for 1D-like circuits
   - Can handle 100+ qubits if entanglement is limited

3. STABILIZER SIMULATION (Gottesman-Knill)
   - Clifford gates only: can simulate exponentially many qubits
   - Shor needs T gates, so this doesn't apply directly

4. PATH INTEGRAL / FEYNMAN SIMULATION
   - Sum over computational paths
   - Good for shallow circuits
   - Cost: O(2^d) where d is depth

PRACTICAL LIMITS
================

Classical simulation (exact):
  - ~45-50 qubits with supercomputers (2024)
  - Memory: 2^n complex numbers × 16 bytes
  - 50 qubits = 16 PB of RAM

Quantum hardware (2024-2025):
  - 1000+ noisy qubits available
  - ~100 error-corrected logical qubits (projected)
  - RSA-2048 needs ~4000-6000 logical qubits

Timeline estimates:
  - RSA-2048 broken: 2035-2040 (optimistic) to never (if quantum computers plateau)
"""
    print(approaches)


if __name__ == "__main__":
    print_scaling_analysis()
    print()
    describe_scaling_approaches()
    print()
    analyze_n21()
