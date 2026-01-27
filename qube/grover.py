"""
Grover's search algorithm implementation.

Grover's algorithm provides quadratic speedup for unstructured search problems.
Given a function f(x) that returns 1 for exactly one value x* (the "needle"),
Grover's algorithm finds x* in O(√N) evaluations instead of O(N).
"""

import numpy as np
from typing import List, Callable, Optional

from .core import (
    applyGate, pushQubit, measureQubit, probQubit,
    reset, get_workspace, set_workspace, TOFFn
)
from .gates import H_gate, X_gate, Z_gate


def zero_phase_oracle(qubits: List[str]):
    """
    Phase oracle that marks the all-zeros state.

    Applies a phase flip (-1) to the |00...0⟩ state only.
    This is used as part of the diffusion operator.

    Args:
        qubits: List of qubit names
    """
    # Flip phase of |0...0⟩: X all qubits, then multi-controlled Z, then X all
    for q in qubits:
        applyGate(X_gate, q)

    # Multi-controlled Z = H on last, multi-controlled X, H on last
    applyGate(H_gate, qubits[-1])
    TOFFn(qubits[:-1], qubits[-1])
    applyGate(H_gate, qubits[-1])

    for q in qubits:
        applyGate(X_gate, q)


def diffusion_operator(qubits: List[str]):
    """
    Grover diffusion operator (inversion about average).

    D = 2|ψ⟩⟨ψ| - I where |ψ⟩ is the uniform superposition.
    This can be implemented as: H⊗n · (2|0⟩⟨0| - I) · H⊗n

    Args:
        qubits: List of qubit names
    """
    # Apply H to all qubits
    for q in qubits:
        applyGate(H_gate, q)

    # Apply phase oracle for |0...0⟩
    zero_phase_oracle(qubits)

    # Apply H to all qubits again
    for q in qubits:
        applyGate(H_gate, q)


def grover_search(
    n: int,
    oracle: Callable[[List[str]], None],
    num_iterations: Optional[int] = None,
    verbose: bool = True
) -> Optional[int]:
    """
    Run Grover's search algorithm.

    Args:
        n: Number of qubits (searches 2^n items)
        oracle: Function that applies phase flip to marked states.
                Takes list of qubit names as argument.
        num_iterations: Number of Grover iterations (default: optimal ~π√N/4)
        verbose: If True, print progress

    Returns:
        Measured result (integer), or None if search failed
    """
    reset()

    # Calculate optimal number of iterations if not specified
    N = 2 ** n
    if num_iterations is None:
        num_iterations = int(np.pi / 4 * np.sqrt(N))

    if verbose:
        print(f"Grover search on {n} qubits ({N} items)")
        print(f"Using {num_iterations} iterations")

    # Initialize qubits in superposition
    qubits = []
    for i in range(n):
        name = f"Q{i}"
        pushQubit(name, [1, 0])
        applyGate(H_gate, name)
        qubits.append(name)

    if verbose:
        print("Initial uniform superposition created")

    # Grover iterations
    for iteration in range(num_iterations):
        # Oracle: mark the target state(s)
        oracle(qubits)

        # Diffusion: inversion about average
        diffusion_operator(qubits)

        if verbose:
            # Show probability of first qubit being 1 as a rough progress indicator
            print(f"Iteration {iteration + 1}: amplification in progress")

    if verbose:
        print("Measuring result...")

    # Measure all qubits
    result = 0
    for i in range(n - 1, -1, -1):
        bit = int(measureQubit(qubits[i]))
        result = result * 2 + bit

    if verbose:
        print(f"Measured: {result}")

    return result


def create_single_target_oracle(target: int, n: int) -> Callable[[List[str]], None]:
    """
    Create an oracle that marks a single target value.

    Args:
        target: The value to search for
        n: Number of qubits

    Returns:
        Oracle function that can be passed to grover_search
    """
    def oracle(qubits: List[str]):
        # Apply X to qubits where target bit is 0
        for i, q in enumerate(qubits):
            if not ((target >> i) & 1):
                applyGate(X_gate, q)

        # Multi-controlled Z
        applyGate(H_gate, qubits[-1])
        TOFFn(qubits[:-1], qubits[-1])
        applyGate(H_gate, qubits[-1])

        # Undo X gates
        for i, q in enumerate(qubits):
            if not ((target >> i) & 1):
                applyGate(X_gate, q)

    return oracle


def grover_search_for_value(n: int, target: int, verbose: bool = True) -> Optional[int]:
    """
    Convenience function to search for a specific value.

    Args:
        n: Number of qubits
        target: Value to search for (0 to 2^n - 1)
        verbose: If True, print progress

    Returns:
        Measured result (should equal target with high probability)
    """
    oracle = create_single_target_oracle(target, n)
    return grover_search(n, oracle, verbose=verbose)
