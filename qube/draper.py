"""
Draper QFT Adder - Addition using the Quantum Fourier Transform.

The Draper adder performs quantum addition by working in the "phase basis"
(Fourier basis) where addition reduces to simple phase rotations. This is
more efficient than ripple-carry adders for certain applications.

Algorithm:
1. Apply QFT to transform the target register into phase basis
2. Apply controlled phase rotations to add the value
3. Apply inverse QFT to return to computational basis

References:
- T. G. Draper, "Addition on a Quantum Computer", 2000. arXiv:quant-ph/0008033
- Ruiz-Perez et al., "Quantum arithmetic with the QFT", 2017. arXiv:1411.5949

Complexity:
- Adding constant to n-qubit register: O(n²) gates
- Adding two n-qubit registers: O(n²) gates
- No ancilla qubits required!
"""

import numpy as np
from typing import List, Optional

from .core import applyGate, pushQubit, measureQubit, reset, get_state
from .gates import P_gate, CP_gate, H_gate, SWAP_gate
from .qft import QFT, QFT_inverse


def phi_add_constant(qubits: List[str], constant: int, inverse: bool = False):
    """
    Add a classical constant to a register already in Fourier basis.

    This is the core operation: given |φ(a)⟩ in phase encoding,
    produces |φ(a + constant)⟩.

    After QFT (which includes swaps to reverse order), the j-th qubit
    receives a phase rotation of:
        θ_j = 2π · constant / 2^(j+1)

    where j is 0-indexed from MSB. So MSB gets largest rotation (π·const),
    LSB gets smallest (2π·const/2^n).

    Args:
        qubits: List of qubit names [MSB, ..., LSB] (already in Fourier basis)
        constant: Classical integer to add
        inverse: If True, subtract instead of add
    """
    n = len(qubits)
    sign = -1 if inverse else 1

    for j in range(n):
        # After QFT's swap, qubit j (0=MSB) has phase contribution from 2^(j+1)
        # So to add constant k, rotate by 2πk/2^(j+1)
        divisor = 2 ** (j + 1)
        theta = sign * 2 * np.pi * constant / divisor
        applyGate(P_gate(theta), qubits[j])


def draper_add_constant(qubits: List[str], constant: int, inverse: bool = False):
    """
    Add a classical constant to a quantum register using QFT.

    Computes: |a⟩ → |a + constant⟩ (mod 2^n)

    This is the simplest form of the Draper adder. It:
    1. Applies QFT to transform to Fourier basis
    2. Adds phase rotations encoding the constant
    3. Applies inverse QFT to return to computational basis

    Args:
        qubits: List of qubit names [MSB, ..., LSB]
        constant: Classical integer to add (can be negative for subtraction)
        inverse: If True, subtract instead of add

    Example:
        # Add 3 to a 4-qubit register initialized to |5⟩
        reset()
        for i, b in enumerate([0, 1, 0, 1]):  # 5 = 0101
            pushQubit(f"a{i}", [1-b, b])
        draper_add_constant(["a0", "a1", "a2", "a3"], 3)
        # Result: register now holds |8⟩ = |1000⟩
    """
    from .core import tosQubit

    # Step 1: Transform to Fourier basis
    QFT(qubits)

    # Normalize qubit order after QFT (QFT may reorder namestack)
    for q in qubits:
        tosQubit(q)

    # Step 2: Add constant via phase rotations
    phi_add_constant(qubits, constant, inverse=inverse)

    # Step 3: Return to computational basis
    QFT_inverse(qubits)

    # Normalize order after inverse QFT
    for q in qubits:
        tosQubit(q)


def phi_add_register(a_qubits: List[str], b_qubits: List[str], inverse: bool = False):
    """
    Add register b to register a, where a is already in Fourier basis.

    Given |φ(a)⟩|b⟩, produces |φ(a + b)⟩|b⟩.

    After QFT's swap, a[j] has phase contribution from 2^(j+1) in the sum.
    When b[k]=1 (meaning 2^(n_b-1-k) contribution), we need to rotate a[j]
    by 2π * 2^(n_b-1-k) / 2^(j+1).

    Args:
        a_qubits: Target register [MSB, ..., LSB] (in Fourier basis)
        b_qubits: Source register [MSB, ..., LSB] (computational basis)
        inverse: If True, subtract b from a
    """
    n_a = len(a_qubits)
    n_b = len(b_qubits)
    sign = -1 if inverse else 1

    # For each target qubit a[j] (j=0 is MSB after QFT swap)
    for j in range(n_a):
        # For each source qubit b[k] (k=0 is MSB)
        for k in range(n_b):
            # b[k] represents value 2^(n_b - 1 - k) when it's |1⟩
            # a[j] after QFT has phase sensitivity 2^(j+1) (due to swap)
            #
            # To add value 2^(n_b-1-k) to a[j], rotate by:
            #   2π * 2^(n_b-1-k) / 2^(j+1)
            # = 2π / 2^(j+1 - (n_b-1-k))
            # = 2π / 2^(j + 2 - n_b + k)
            # = 2π / 2^(j - n_b + k + 2)

            exponent = j - n_b + k + 2

            # Only apply if exponent > 0 (otherwise rotation >= 2π)
            if exponent > 0:
                theta = sign * 2 * np.pi / (2 ** exponent)
                applyGate(CP_gate(theta), b_qubits[k], a_qubits[j])


def draper_add(a_qubits: List[str], b_qubits: List[str], inverse: bool = False):
    """
    Add register b to register a using QFT (result in a, b unchanged).

    Computes: |a⟩|b⟩ → |a + b⟩|b⟩ (mod 2^n where n = len(a))

    This is the full Draper adder for two quantum registers. It:
    1. Applies QFT to register a
    2. Applies controlled phase rotations from b to a
    3. Applies inverse QFT to register a

    The register b is not modified (it remains in computational basis).

    Args:
        a_qubits: Target register [MSB, ..., LSB], receives the sum
        b_qubits: Source register [MSB, ..., LSB], unchanged
        inverse: If True, subtract b from a

    Example:
        # Compute 5 + 3 = 8
        reset()
        # Register a = |5⟩ = |0101⟩
        for i, b in enumerate([0, 1, 0, 1]):
            pushQubit(f"a{i}", [1-b, b])
        # Register b = |3⟩ = |0011⟩
        for i, b in enumerate([0, 0, 1, 1]):
            pushQubit(f"b{i}", [1-b, b])

        draper_add(["a0", "a1", "a2", "a3"], ["b0", "b1", "b2", "b3"])
        # Result: a = |8⟩ = |1000⟩, b = |3⟩ unchanged
    """
    from .core import tosQubit

    # Step 1: Transform a to Fourier basis
    QFT(a_qubits)

    # Normalize qubit order after QFT
    for q in a_qubits + b_qubits:
        tosQubit(q)

    # Step 2: Add b via controlled phase rotations
    phi_add_register(a_qubits, b_qubits, inverse=inverse)

    # Step 3: Return a to computational basis
    QFT_inverse(a_qubits)

    # Normalize order after inverse QFT
    for q in a_qubits + b_qubits:
        tosQubit(q)


def draper_subtract(a_qubits: List[str], b_qubits: List[str]):
    """
    Subtract register b from register a: |a⟩|b⟩ → |a - b⟩|b⟩.

    Convenience wrapper for draper_add with inverse=True.
    Result is modulo 2^n (wraps around for negative results).
    """
    draper_add(a_qubits, b_qubits, inverse=True)


def draper_subtract_constant(qubits: List[str], constant: int):
    """
    Subtract a classical constant from a register: |a⟩ → |a - constant⟩.

    Convenience wrapper for draper_add_constant with inverse=True.
    """
    draper_add_constant(qubits, constant, inverse=True)


# =============================================================================
# Testing utilities
# =============================================================================

def measure_register(qubits: List[str]) -> int:
    """
    Measure a register and return the integer value.

    Args:
        qubits: List of qubit names [MSB, ..., LSB]

    Returns:
        Integer value of the measured state
    """
    from .core import tosQubit

    # Normalize qubit order first
    for q in qubits:
        tosQubit(q)

    result = 0
    for i, q in enumerate(qubits):
        bit = int(measureQubit(q))
        result = result * 2 + bit  # MSB first
    return result


def init_register(prefix: str, value: int, n_bits: int) -> List[str]:
    """
    Initialize a quantum register to a classical value.

    Args:
        prefix: Name prefix for qubits
        value: Classical integer value
        n_bits: Number of qubits

    Returns:
        List of qubit names [MSB, ..., LSB]
    """
    qubits = []
    for i in range(n_bits):
        bit = (value >> (n_bits - 1 - i)) & 1  # MSB first
        pushQubit(f"{prefix}{i}", [1 - bit, bit])
        qubits.append(f"{prefix}{i}")
    return qubits


def test_draper_add_constant():
    """Test adding classical constants to quantum registers."""
    print("Testing draper_add_constant:")

    test_cases = [
        (4, 0, 3, 3),    # 0 + 3 = 3
        (4, 5, 3, 8),    # 5 + 3 = 8
        (4, 7, 1, 8),    # 7 + 1 = 8
        (4, 15, 1, 0),   # 15 + 1 = 0 (overflow)
        (4, 10, 7, 1),   # 10 + 7 = 17 mod 16 = 1
        (4, 5, 0, 5),    # 5 + 0 = 5
    ]

    all_passed = True
    for n_bits, a, const, expected in test_cases:
        reset()
        qubits = init_register("a", a, n_bits)
        draper_add_constant(qubits, const)

        # Normalize order before measuring
        from .core import tosQubit
        for q in qubits:
            tosQubit(q)

        result = measure_register(qubits)
        status = "✓" if result == expected else "✗"
        print(f"  {a} + {const} mod {2**n_bits} = {result} (expected {expected}) {status}")
        if result != expected:
            all_passed = False

    return all_passed


def test_draper_add():
    """Test adding two quantum registers."""
    print("\nTesting draper_add:")

    test_cases = [
        (4, 5, 3, 8, 3),    # 5 + 3 = 8, b unchanged
        (4, 0, 7, 7, 7),    # 0 + 7 = 7
        (4, 15, 1, 0, 1),   # 15 + 1 = 0 (overflow)
        (4, 6, 6, 12, 6),   # 6 + 6 = 12
        (4, 10, 10, 4, 10), # 10 + 10 = 20 mod 16 = 4
    ]

    all_passed = True
    for n_bits, a, b, expected_a, expected_b in test_cases:
        reset()
        a_qubits = init_register("a", a, n_bits)
        b_qubits = init_register("b", b, n_bits)

        draper_add(a_qubits, b_qubits)

        # Normalize order before measuring
        from .core import tosQubit
        for q in a_qubits + b_qubits:
            tosQubit(q)

        result_a = measure_register(a_qubits)
        result_b = measure_register(b_qubits)

        status_a = "✓" if result_a == expected_a else "✗"
        status_b = "✓" if result_b == expected_b else "✗"

        print(f"  |{a}⟩|{b}⟩ → |{result_a}⟩|{result_b}⟩ "
              f"(expected |{expected_a}⟩|{expected_b}⟩) {status_a}{status_b}")

        if result_a != expected_a or result_b != expected_b:
            all_passed = False

    return all_passed


def test_draper_subtract():
    """Test subtraction."""
    print("\nTesting draper_subtract:")

    test_cases = [
        (4, 8, 3, 5),     # 8 - 3 = 5
        (4, 5, 5, 0),     # 5 - 5 = 0
        (4, 3, 5, 14),    # 3 - 5 = -2 mod 16 = 14
        (4, 0, 1, 15),    # 0 - 1 = -1 mod 16 = 15
    ]

    all_passed = True
    for n_bits, a, b, expected in test_cases:
        reset()
        a_qubits = init_register("a", a, n_bits)
        b_qubits = init_register("b", b, n_bits)

        draper_subtract(a_qubits, b_qubits)

        from .core import tosQubit
        for q in a_qubits:
            tosQubit(q)

        result = measure_register(a_qubits)
        status = "✓" if result == expected else "✗"
        print(f"  {a} - {b} mod {2**n_bits} = {result} (expected {expected}) {status}")

        if result != expected:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("DRAPER QFT ADDER TESTS")
    print("=" * 60)
    print()

    all_passed = True
    all_passed &= test_draper_add_constant()
    all_passed &= test_draper_add()
    all_passed &= test_draper_subtract()

    print()
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
