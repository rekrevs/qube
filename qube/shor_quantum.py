"""
Generalized Quantum Shor's Algorithm.

This module implements Shor's factoring algorithm for arbitrary N,
using permutation cycles for controlled modular multiplication.

Key components:
- Controlled modular multiplication via permutation cycles
- Full Shor's algorithm with quantum phase estimation

References:
- Beauregard, "Circuit for Shor's algorithm using 2n+3 qubits", 2002
"""

import math
import random
from typing import List, Tuple, Optional

from .core import applyGate, pushQubit, measureQubit, reset, tosQubit
from .gates import H_gate, X_gate, CNOT_gate, TOFF_gate
from .qft import QFT_inverse
from .draper import init_register
from .utils import gcd, is_coprime, extract_period


# =============================================================================
# Register sizing
# =============================================================================

def compute_register_sizes(N: int) -> Tuple[int, int]:
    """
    Compute the required register sizes for factoring N.

    Args:
        N: Number to factor

    Returns:
        Tuple (work_size, control_size) where:
        - work_size = ceil(log2(N)) bits to hold values 0..N-1
        - control_size = 2 * work_size for sufficient precision in phase estimation
    """
    work_size = math.ceil(math.log2(N))
    control_size = 2 * work_size
    return (work_size, control_size)


# =============================================================================
# Controlled modular multiplication
# =============================================================================

def find_multiplication_cycles(a: int, N: int) -> List[List[int]]:
    """
    Find cycles in the permutation x -> a*x mod N.

    For elements in the multiplicative group mod N, multiplication by a
    forms a permutation that can be decomposed into disjoint cycles.

    Args:
        a: Multiplier (must be coprime to N)
        N: Modulus

    Returns:
        List of cycles, where each cycle is a list of integers
    """
    if gcd(a, N) != 1:
        raise ValueError(f"{a} and {N} must be coprime")

    visited = set()
    cycles = []

    # Only consider elements coprime to N (the multiplicative group)
    for start in range(1, N):
        if start in visited or gcd(start, N) != 1:
            continue

        # Build cycle starting from 'start'
        cycle = []
        x = start
        while x not in visited:
            visited.add(x)
            cycle.append(x)
            x = (x * a) % N

        if len(cycle) > 1:
            cycles.append(cycle)

    return cycles


def controlled_swap_states(control: str, qubits: List[str], state_a: int, state_b: int):
    """
    Controlled swap of two computational basis states.

    When control is |1>, swaps |state_a> <-> |state_b> in the work register.
    All other basis states are left unchanged.

    Implementation: Use linear path through intermediate states.
    For states a and b differing in k bits, we:
    1. Walk from a toward b, flipping one bit at a time: a -> s1 -> s2 -> ... -> b
    2. Walk back from b to a (except the last step): b -> ... -> s2 -> s1
    This gives a total of 2k-1 single-bit swaps.

    The effect is: a ends up at b's position, and b ends up at a's position.

    Args:
        control: Control qubit name
        qubits: List of qubit names [MSB, ..., LSB]
        state_a, state_b: The two basis states to swap (integers)
    """
    n = len(qubits)

    if state_a == state_b:
        return  # States are identical, nothing to do

    # Find which bits differ
    diff = state_a ^ state_b
    diff_positions = [i for i in range(n) if (diff >> (n - 1 - i)) & 1]

    # Simple case: single bit differs
    if len(diff_positions) == 1:
        _swap_single_bit_diff(control, qubits, state_a, diff_positions[0])
        return

    # Multi-bit case: walk along Hamming path
    # Build path: a = p[0], p[1], ..., p[k] = b
    # where p[i] and p[i+1] differ in exactly one bit
    path = [state_a]
    current = state_a
    for pos in diff_positions:
        current ^= (1 << (n - 1 - pos))
        path.append(current)
    # path[-1] should equal state_b

    # Forward pass: swap p[0]<->p[1], p[1]<->p[2], ..., p[k-1]<->p[k]
    # After this: original |a> is at position p[k]=b, original |b> is at p[k-1]
    for i, pos in enumerate(diff_positions):
        _swap_single_bit_diff(control, qubits, path[i], pos)

    # Backward pass: swap p[k-1]<->p[k-2], ..., p[1]<->p[0]
    # This moves |b> from p[k-1] back to p[0]=a
    for i in range(len(diff_positions) - 2, -1, -1):
        _swap_single_bit_diff(control, qubits, path[i], diff_positions[i])


def _swap_single_bit_diff(control: str, qubits: List[str], state_a: int, diff_pos: int):
    """
    Swap states a and b where they differ in exactly one bit (diff_pos).

    Control on external control + all other bits matching state_a's pattern.
    This uniquely identifies {a, b} because any state matching those bits
    must be either a or b.
    """
    n = len(qubits)

    # Build controls: external control + all positions except diff_pos
    controls = [control]
    for pos in range(n):
        if pos == diff_pos:
            continue  # Skip the differing position
        bit_val = (state_a >> (n - 1 - pos)) & 1
        if bit_val == 0:
            applyGate(X_gate, qubits[pos])
        controls.append(qubits[pos])

    # Apply multi-controlled X to the differing bit
    _apply_multi_controlled_x(controls, qubits[diff_pos])

    # Unflip
    for pos in range(n):
        if pos == diff_pos:
            continue
        bit_val = (state_a >> (n - 1 - pos)) & 1
        if bit_val == 0:
            applyGate(X_gate, qubits[pos])


def _apply_multi_controlled_x(controls: List[str], target: str):
    """Apply X gate on target controlled by all qubits in controls list."""
    if len(controls) == 1:
        applyGate(CNOT_gate, controls[0], target)
    elif len(controls) == 2:
        applyGate(TOFF_gate, controls[0], controls[1], target)
    else:
        # Use ancilla for cascading
        from .core import TOFFn
        TOFFn(controls, target)


def controlled_mod_mult(control: str, work_qubits: List[str], a: int, N: int):
    """
    Controlled modular multiplication: |ctrl>|x> -> |ctrl>|a*x mod N> if ctrl=1.

    When control is |0>, the work register is unchanged.
    When control is |1>, applies the permutation x -> a*x mod N.

    The permutation is implemented by decomposing into cycles and applying
    controlled transpositions for each cycle.

    Args:
        control: Control qubit name
        work_qubits: List of work qubit names [MSB, ..., LSB]
        a: Multiplier (must be coprime to N)
        N: Modulus
    """
    # Find the cycle structure of multiplication by a mod N
    cycles = find_multiplication_cycles(a, N)

    # Apply each cycle as a sequence of transpositions
    # A cycle (c0, c1, c2, ..., ck) can be implemented as:
    # swap(c0, c1), swap(c0, c2), ..., swap(c0, ck)
    for cycle in cycles:
        if len(cycle) <= 1:
            continue

        # Apply transpositions: (c0, c1), (c0, c2), ..., (c0, c_{k-1})
        for i in range(1, len(cycle)):
            controlled_swap_states(control, work_qubits, cycle[0], cycle[i])


# =============================================================================
# Controlled modular exponentiation
# =============================================================================

def apply_controlled_mod_exp(control_qubits: List[str], work_qubits: List[str],
                              a: int, N: int):
    """
    Apply controlled modular exponentiation for Shor's algorithm.

    For each control qubit k, applies controlled multiplication by a^(2^k) mod N.
    This builds up the transformation:
        |j>|1> -> |j>|a^j mod N>

    where j is the integer represented by the control register.

    Args:
        control_qubits: List of control qubit names (LSB to MSB order, i.e., c0 controls a^1)
        work_qubits: List of work qubit names [MSB, ..., LSB]
        a: Base for exponentiation
        N: Modulus
    """
    # For each control qubit k, apply controlled multiplication by a^(2^k)
    power = a
    for _k, ctrl in enumerate(control_qubits):
        controlled_mod_mult(ctrl, work_qubits, power, N)
        power = (power * power) % N  # a^(2^(k+1)) = (a^(2^k))^2


# =============================================================================
# Full Shor's algorithm
# =============================================================================

def shor_factor(N: int, a: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """
    Shor's algorithm to factor N using quantum period finding.

    Args:
        N: Number to factor (should be composite, odd, and not a prime power)
        a: Base for modular exponentiation (random if not specified)

    Returns:
        Tuple (factor1, factor2) if successful, or None if this run failed
    """
    # Classical preprocessing
    if N % 2 == 0:
        return (2, N // 2)

    # Check if N is a prime power
    for b in range(2, int(math.log2(N)) + 1):
        root = int(round(N ** (1 / b)))
        for candidate_root in [root - 1, root, root + 1]:
            if candidate_root > 1 and candidate_root ** b == N:
                return (candidate_root, N // candidate_root)

    # Choose random a if not specified
    if a is None:
        candidates = [x for x in range(2, N) if is_coprime(x, N)]
        if not candidates:
            return None
        a = random.choice(candidates)

    # Check for lucky GCD
    g = gcd(a, N)
    if g > 1:
        return (g, N // g)

    # Quantum period finding
    reset()

    work_size, control_size = compute_register_sizes(N)

    # We need one extra bit in work register for valid range
    # But for the cycle-based multiplication, we work with values < N
    # So work_size bits suffice for values 0..N-1
    n_work = work_size

    # Initialize work register to |1>
    work_qubits = init_register("w", 1, n_bits=n_work)

    # Initialize control register in superposition
    control_qubits = []
    for i in range(control_size):
        name = f"c{i}"
        pushQubit(name, [1, 0])
        applyGate(H_gate, name)
        control_qubits.append(name)

    # Apply controlled modular exponentiation
    apply_controlled_mod_exp(control_qubits, work_qubits, a, N)

    # Apply inverse QFT to control register
    # QFT expects MSB first, but our control_qubits are LSB first
    QFT_inverse(control_qubits[::-1])

    # Measure control register
    # Normalize order first
    for q in control_qubits[::-1]:
        tosQubit(q)

    measurement = 0
    for i in range(control_size - 1, -1, -1):
        bit = int(measureQubit(control_qubits[i]))
        measurement = measurement * 2 + bit

    # Clean up work register
    for q in reversed(work_qubits):
        measureQubit(q)

    # Classical post-processing: extract period
    r: Optional[int] = extract_period(measurement, control_size, N, a)

    if r is None or r % 2 == 1:
        return None

    # Compute factors
    x = pow(a, r // 2, N)

    if x == N - 1:  # x = -1 mod N
        return None

    factor1 = gcd(x - 1, N)
    factor2 = gcd(x + 1, N)

    if factor1 != 1 and factor1 != N:
        return (factor1, N // factor1)
    elif factor2 != 1 and factor2 != N:
        return (factor2, N // factor2)
    else:
        return None
