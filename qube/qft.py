"""
Quantum Fourier Transform (QFT) implementation.

The QFT is the quantum analog of the discrete Fourier transform and is a key
component of many quantum algorithms including Shor's factoring algorithm
and quantum phase estimation.
"""

import numpy as np
from typing import List

from .core import applyGate, pushQubit, measureQubit
from .gates import H_gate, SWAP_gate, CP_gate, P_gate, CNOT_gate


def QFT(qubits: List[str], verbose: bool = False):
    """
    Quantum Fourier Transform on a list of qubits.

    The QFT transforms the computational basis states as:
    |j⟩ → (1/√N) Σₖ exp(2πijk/N) |k⟩

    Args:
        qubits: List of qubit names, ordered from most significant to least
                significant bit (MSB first, LSB last)
        verbose: If True, print state after each step
    """
    n = len(qubits)

    if verbose:
        print(f"QFT on {n} qubits: {qubits}")

    # Apply Hadamard and controlled rotations
    for i in range(n):
        applyGate(H_gate, qubits[i])

        if verbose:
            print(f"After H on {qubits[i]}")

        # Controlled phase rotations
        for j in range(i + 1, n):
            # Rotation angle: π/2^(j-i)
            theta = np.pi / (2 ** (j - i))
            applyGate(CP_gate(theta), qubits[j], qubits[i])

            if verbose:
                print(f"After CP(π/{2 ** (j - i)}) controlled by {qubits[j]} on {qubits[i]}")

    # Swap qubits to reverse order
    for i in range(n // 2):
        applyGate(SWAP_gate, qubits[i], qubits[n - 1 - i])
        if verbose:
            print(f"After SWAP({qubits[i]}, {qubits[n - 1 - i]})")

    if verbose:
        print("QFT complete")


def QFT_inverse(qubits: List[str], verbose: bool = False):
    """
    Inverse Quantum Fourier Transform.

    The inverse QFT is the adjoint of QFT, obtained by reversing the gate
    order and negating the phase angles.

    Args:
        qubits: List of qubit names, ordered MSB to LSB
        verbose: If True, print state after each step
    """
    n = len(qubits)

    if verbose:
        print(f"Inverse QFT on {n} qubits: {qubits}")

    # First, swap qubits to reverse order (same as forward QFT)
    for i in range(n // 2):
        applyGate(SWAP_gate, qubits[i], qubits[n - 1 - i])

    # Apply gates in reverse order with negated phases
    for i in range(n - 1, -1, -1):
        # Controlled phase rotations (in reverse order, with negative angles)
        for j in range(n - 1, i, -1):
            theta = -np.pi / (2 ** (j - i))  # Negative angle for inverse
            applyGate(CP_gate(theta), qubits[j], qubits[i])

        # Hadamard on qubit i (H is its own inverse)
        applyGate(H_gate, qubits[i])

    if verbose:
        print("Inverse QFT complete")


def QFT_decomposed(qubits: List[str], verbose: bool = False) -> int:
    """
    QFT using only basic gates (H, CNOT, P).

    All controlled-phase gates are decomposed into single-qubit and CNOT gates.
    This is useful for understanding the gate count and for hardware that
    doesn't natively support controlled-phase gates.

    Args:
        qubits: List of qubit names, ordered MSB to LSB
        verbose: If True, print gate count

    Returns:
        Total gate count
    """
    n = len(qubits)
    gate_count = 0

    for i in range(n):
        applyGate(H_gate, qubits[i])
        gate_count += 1

        for j in range(i + 1, n):
            theta = np.pi / (2 ** (j - i))
            # Decomposed CP: P(θ/2), CNOT, P(-θ/2), CNOT, P(θ/2)
            applyGate(P_gate(theta / 2), qubits[i])
            applyGate(CNOT_gate, qubits[j], qubits[i])
            applyGate(P_gate(-theta / 2), qubits[i])
            applyGate(CNOT_gate, qubits[j], qubits[i])
            applyGate(P_gate(theta / 2), qubits[j])
            gate_count += 5

    # Swaps (each SWAP = 3 CNOTs)
    for i in range(n // 2):
        applyGate(SWAP_gate, qubits[i], qubits[n - 1 - i])
        gate_count += 1  # Count as 1, but it's 3 CNOTs if decomposed

    if verbose:
        print(f"QFT_decomposed used {gate_count} gates")

    return gate_count


def QFT_inverse_decomposed(qubits: List[str], verbose: bool = False) -> int:
    """
    Inverse QFT using only basic gates.

    Args:
        qubits: List of qubit names, ordered MSB to LSB
        verbose: If True, print gate count

    Returns:
        Total gate count
    """
    n = len(qubits)
    gate_count = 0

    # Swaps first
    for i in range(n // 2):
        applyGate(SWAP_gate, qubits[i], qubits[n - 1 - i])
        gate_count += 1

    # Inverse gates
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            theta = -np.pi / (2 ** (j - i))
            applyGate(P_gate(theta / 2), qubits[i])
            applyGate(CNOT_gate, qubits[j], qubits[i])
            applyGate(P_gate(-theta / 2), qubits[i])
            applyGate(CNOT_gate, qubits[j], qubits[i])
            applyGate(P_gate(theta / 2), qubits[j])
            gate_count += 5

        applyGate(H_gate, qubits[i])
        gate_count += 1

    if verbose:
        print(f"QFT_inverse_decomposed used {gate_count} gates")

    return gate_count
