"""
Quantum gate definitions.

This module contains all standard quantum gates used in quantum computing,
including single-qubit gates (Pauli, Hadamard, phase, rotation) and
multi-qubit gates (CNOT, CZ, SWAP, Toffoli, controlled-phase).
"""

import numpy as np

# =============================================================================
# Single-qubit gates
# =============================================================================

X_gate = np.array([[0, 1],      # Pauli X gate (NOT gate)
                   [1, 0]])

Y_gate = np.array([[ 0, -1j],   # Pauli Y gate
                   [1j,   0]])

Z_gate = np.array([[1,  0],     # Pauli Z gate = P(π) = S²
                   [0, -1]])

H_gate = np.array([[1,  1],     # Hadamard gate
                   [1, -1]]) * np.sqrt(1/2)

S_gate = np.array([[1,  0],     # Phase gate = P(π/2) = T²
                   [0, 1j]])

T_gate = np.array([[1,                  0],   # T gate = P(π/4)
                   [0, np.exp(np.pi / -4j)]])

Tinv_gate = np.array([[1,                 0],   # T† gate = P(-π/4)
                      [0, np.exp(np.pi / 4j)]])

I_gate = np.array([[1, 0],      # Identity gate
                   [0, 1]])


def P_gate(phi):
    """Phase shift gate P(φ) = diag(1, e^{iφ})"""
    return np.array([[1,              0],
                     [0, np.exp(phi * 1j)]])


def Rx_gate(theta):
    """X rotation gate Rx(θ)"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c,    -1j * s],
                     [-1j * s,    c]])


def Ry_gate(theta):
    """Y rotation gate Ry(θ)"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s],
                     [s,  c]])


def Rz_gate(theta):
    """Z rotation gate Rz(θ)"""
    return np.array([[np.exp(-1j * theta / 2),                    0],
                     [                      0, np.exp(1j * theta / 2)]])


# =============================================================================
# Two-qubit gates
# =============================================================================

CNOT_gate = np.array([[1, 0, 0, 0],   # Controlled NOT gate (XOR)
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]])

CZ_gate = np.array([[1, 0, 0,  0],    # Controlled Z gate
                    [0, 1, 0,  0],
                    [0, 0, 1,  0],
                    [0, 0, 0, -1]])

SWAP_gate = np.array([[1, 0, 0, 0],   # Swap gate
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])


def CP_gate(theta):
    """Controlled phase gate CP(θ) = diag(1, 1, 1, e^{iθ})"""
    return np.array([[1, 0, 0,               0],
                     [0, 1, 0,               0],
                     [0, 0, 1,               0],
                     [0, 0, 0, np.exp(1j * theta)]])


# =============================================================================
# Three-qubit gates
# =============================================================================

TOFF_gate = np.array([[1, 0, 0, 0, 0, 0, 0, 0],   # Toffoli gate (CCNOT)
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 1, 0]])
