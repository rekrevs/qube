"""
Core quantum simulation functionality.

This module provides the fundamental operations for simulating quantum circuits
using a state-vector simulator with a stack-based qubit management system.

The workspace is a numpy array representing the quantum state, and qubits are
managed via a name stack that allows referencing qubits by name.
"""

import numpy as np
from typing import List, Tuple, Optional

# Global state
workspace: np.ndarray = np.array([[1.0 + 0j]])
namestack: List[str] = []


def reset():
    """Reset the quantum workspace to empty state."""
    global workspace, namestack
    workspace = np.array([[1.0 + 0j]])
    namestack = []


def get_state() -> np.ndarray:
    """Return a copy of the current quantum state as a flat vector."""
    return np.reshape(workspace, -1).copy()


def get_workspace() -> np.ndarray:
    """Return a reference to the workspace (for advanced use)."""
    global workspace
    return workspace


def set_workspace(ws: np.ndarray):
    """Set the workspace directly (for advanced use)."""
    global workspace
    workspace = ws


def get_namestack() -> List[str]:
    """Return a copy of the current name stack."""
    return namestack.copy()


def pushQubit(name: str, weights: List[float]):
    """
    Push a new qubit onto the workspace.

    Args:
        name: Unique name for the qubit
        weights: Initial amplitudes [|0⟩ amplitude, |1⟩ amplitude]
                 Will be automatically normalized.
    """
    global workspace, namestack

    if workspace.shape == (1, 1) and abs(workspace[0, 0] - 1.0) < 1e-10:
        # Workspace is empty, reset name stack
        namestack = []

    namestack.append(name)
    weights = np.array(weights, dtype=complex)
    weights = weights / np.linalg.norm(weights)  # Normalize
    workspace = np.reshape(workspace, (1, -1))
    workspace = np.kron(workspace, weights)


def tosQubit(name: str):
    """
    Move the named qubit to the top of stack (TOS).

    This is done by swapping axes in the workspace array.

    Args:
        name: Name of the qubit to move to TOS
    """
    global workspace, namestack

    k = len(namestack) - namestack.index(name)  # Position from TOS
    if k > 1:
        namestack.append(namestack.pop(-k))  # Rotate name stack
        workspace = np.reshape(workspace, (-1, 2, 2 ** (k - 1)))
        workspace = np.swapaxes(workspace, -2, -1)


def applyGate(gate: np.ndarray, *names: str):
    """
    Apply a quantum gate to the specified qubit(s).

    Args:
        gate: The gate matrix (2x2 for single qubit, 4x4 for two qubits, etc.)
        *names: Names of the qubits to apply the gate to (in order)
    """
    global workspace

    if len(names) != len(set(names)):
        raise ValueError("The same qubit cannot occur twice as an argument")

    for name in names:
        tosQubit(name)

    workspace = np.reshape(workspace, (-1, gate.shape[0]))
    np.matmul(workspace, gate.T, out=workspace)


def probQubit(name: str) -> np.ndarray:
    """
    Get the probabilities of measuring the qubit as |0⟩ or |1⟩.

    Args:
        name: Name of the qubit

    Returns:
        Array [P(|0⟩), P(|1⟩)]
    """
    global workspace

    tosQubit(name)
    workspace = np.reshape(workspace, (-1, 2))
    prob = np.linalg.norm(workspace, axis=0) ** 2
    return prob / prob.sum()


def measureQubit(name: str) -> str:
    """
    Measure and remove a qubit from the workspace.

    The qubit collapses to |0⟩ or |1⟩ based on its probabilities,
    and is then removed from the workspace.

    Args:
        name: Name of the qubit to measure

    Returns:
        "0" or "1" indicating the measurement result
    """
    global workspace, namestack

    prob = probQubit(name)
    measurement = np.random.choice(2, p=prob)
    workspace = workspace[:, [measurement]] / np.sqrt(prob[measurement])
    namestack.pop()
    return str(measurement)


def peekQubit(name: str) -> Tuple[float, float]:
    """
    Get the probabilities without collapsing the qubit.

    Args:
        name: Name of the qubit

    Returns:
        Tuple (P(|0⟩), P(|1⟩))
    """
    prob = probQubit(name)
    return (prob[0], prob[1])


# =============================================================================
# Higher-level gate operations using decomposition
# =============================================================================

def toffoli_decomposed(q1: str, q2: str, q3: str):
    """
    Implement Toffoli gate using H, T, T†, and CNOT gates.

    This decomposition uses only gates from a universal gate set.
    """
    from .gates import H_gate, T_gate, Tinv_gate, CNOT_gate

    applyGate(H_gate, q3)
    applyGate(CNOT_gate, q2, q3)
    applyGate(Tinv_gate, q3)
    applyGate(CNOT_gate, q1, q3)
    applyGate(T_gate, q3)
    applyGate(CNOT_gate, q2, q3)
    applyGate(Tinv_gate, q3)
    applyGate(CNOT_gate, q1, q3)
    applyGate(T_gate, q2)
    applyGate(T_gate, q3)
    applyGate(H_gate, q3)
    applyGate(CNOT_gate, q1, q2)
    applyGate(T_gate, q1)
    applyGate(Tinv_gate, q2)
    applyGate(CNOT_gate, q1, q2)


def CP_decomposed(theta: float, control: str, target: str):
    """
    Decompose controlled-phase gate into basic gates.

    CP(θ) = P(θ/2) on target, CNOT, P(-θ/2) on target, CNOT, P(θ/2) on control

    This is equivalent to diag(1, 1, 1, e^{iθ}) up to global phase.
    """
    from .gates import P_gate, CNOT_gate

    applyGate(P_gate(theta / 2), target)
    applyGate(CNOT_gate, control, target)
    applyGate(P_gate(-theta / 2), target)
    applyGate(CNOT_gate, control, target)
    applyGate(P_gate(theta / 2), control)


def TOFF3(q1: str, q2: str, q3: str, q4: str):
    """
    Four-qubit Toffoli: q4 = q4 XOR (q1 AND q2 AND q3).

    Uses an ancilla qubit internally.
    """
    from .gates import TOFF_gate

    pushQubit("_temp", [1, 0])
    applyGate(TOFF_gate, q1, q2, "_temp")
    applyGate(TOFF_gate, "_temp", q3, q4)
    applyGate(TOFF_gate, q1, q2, "_temp")  # Uncompute ancilla
    measureQubit("_temp")


def TOFFn(controls: List[str], result: str):
    """
    Multi-controlled Toffoli: result = result XOR AND(controls).

    Uses ancilla qubits for cascading.
    """
    from .gates import TOFF_gate

    if len(controls) == 1:
        from .gates import CNOT_gate
        applyGate(CNOT_gate, controls[0], result)
        return
    if len(controls) == 2:
        applyGate(TOFF_gate, controls[0], controls[1], result)
        return

    # For 3+ controls, use cascading with ancillas
    ancillas = []
    pushQubit("_anc0", [1, 0])
    ancillas.append("_anc0")
    applyGate(TOFF_gate, controls[0], controls[1], "_anc0")

    for i in range(2, len(controls)):
        anc_name = f"_anc{i - 1}"
        pushQubit(anc_name, [1, 0])
        ancillas.append(anc_name)
        prev_anc = ancillas[-2]
        applyGate(TOFF_gate, prev_anc, controls[i], anc_name)

    # Final gate to result
    from .gates import CNOT_gate
    applyGate(CNOT_gate, ancillas[-1], result)

    # Uncompute ancillas in reverse order
    for i in range(len(controls) - 1, 1, -1):
        prev_anc = ancillas[i - 2] if i > 2 else ancillas[0]
        applyGate(TOFF_gate, prev_anc, controls[i], ancillas[i - 1])

    applyGate(TOFF_gate, controls[0], controls[1], "_anc0")

    # Clean up ancillas
    for anc in reversed(ancillas):
        measureQubit(anc)
