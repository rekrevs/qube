"""Tests for Quantum Fourier Transform."""

import numpy as np
import pytest

from qube import (
    reset, pushQubit, measureQubit, get_state,
    QFT, QFT_inverse, QFT_decomposed,
    allclose_up_to_global_phase,
)


class TestQFT:
    """Tests for QFT implementation."""

    def test_qft_zero_state_gives_uniform_superposition(self):
        """QFT(|00⟩) should give equal superposition."""
        reset()
        pushQubit("Q0", [1, 0])
        pushQubit("Q1", [1, 0])
        QFT(["Q0", "Q1"])

        state = get_state()
        # All 4 amplitudes should be equal (1/2 each)
        expected_probs = np.ones(4) / 4
        assert np.allclose(np.abs(state) ** 2, expected_probs)

    def test_qft_inverse_is_inverse(self):
        """QFT⁻¹(QFT(|ψ⟩)) = |ψ⟩."""
        from qube.core import tosQubit

        reset()
        pushQubit("Q0", [0.6, 0.8])  # Some arbitrary state
        pushQubit("Q1", [0.8, 0.6])
        original = get_state().copy()

        QFT(["Q0", "Q1"])
        QFT_inverse(["Q0", "Q1"])

        # Normalize qubit order (QFT/inverse may reorder namestack)
        tosQubit("Q0")
        tosQubit("Q1")
        state = get_state()
        # Should match original up to global phase
        assert allclose_up_to_global_phase(state, original)

    def test_single_qubit_qft(self):
        """Single qubit QFT(|1⟩) = |−⟩."""
        reset()
        pushQubit("Q", [0, 1])  # |1⟩
        QFT(["Q"])

        state = get_state()
        expected = np.array([1, -1]) / np.sqrt(2)  # |−⟩ state
        assert np.allclose(state, expected)

    def test_three_qubit_qft(self):
        """QFT(|001⟩) on 3 qubits should give correct phases."""
        from qube.core import tosQubit

        reset()
        pushQubit("Q0", [1, 0])  # MSB
        pushQubit("Q1", [1, 0])
        pushQubit("Q2", [0, 1])  # LSB = 1, so state is |001⟩ = 1 in decimal

        QFT(["Q0", "Q1", "Q2"])

        # Normalize qubit order (QFT may reorder namestack)
        tosQubit("Q0")
        tosQubit("Q1")
        tosQubit("Q2")

        state = get_state()
        # QFT|1⟩ for 3 qubits: (1/√8) Σ_k exp(2πik/8)|k⟩
        j = 1  # Input state |001⟩ = 1
        expected = np.array([np.exp(2j * np.pi * j * k / 8) for k in range(8)]) / np.sqrt(8)
        assert np.allclose(state, expected)

    def test_qft_decomposed_matches_matrix(self):
        """QFT_decomposed should match QFT."""
        reset()
        pushQubit("A0", [0.5, 0.866])
        pushQubit("A1", [0.707, 0.707])
        pushQubit("A2", [0.8, 0.6])
        QFT(["A0", "A1", "A2"])
        state_matrix = get_state().copy()
        measureQubit("A2")
        measureQubit("A1")
        measureQubit("A0")

        reset()
        pushQubit("A0", [0.5, 0.866])
        pushQubit("A1", [0.707, 0.707])
        pushQubit("A2", [0.8, 0.6])
        QFT_decomposed(["A0", "A1", "A2"])
        state_decomp = get_state().copy()

        assert np.allclose(state_matrix, state_decomp)


class TestQFTProperties:
    """Tests for QFT mathematical properties."""

    def test_qft_is_unitary(self):
        """QFT should preserve norm."""
        reset()
        pushQubit("Q0", [0.3, 0.9539])  # Arbitrary normalized state
        pushQubit("Q1", [0.6, 0.8])

        original_norm = np.linalg.norm(get_state())
        QFT(["Q0", "Q1"])
        final_norm = np.linalg.norm(get_state())

        assert np.isclose(original_norm, final_norm)

    def test_qft_of_computational_basis(self):
        """QFT of computational basis states should have equal amplitudes."""
        for j in range(4):
            reset()
            # Create |j⟩ state
            pushQubit("Q0", [1 - (j >> 0) & 1, (j >> 0) & 1])
            pushQubit("Q1", [1 - (j >> 1) & 1, (j >> 1) & 1])

            QFT(["Q0", "Q1"])
            state = get_state()

            # All amplitudes should have magnitude 1/2
            assert np.allclose(np.abs(state), 0.5)

    def test_qft_phase_pattern(self):
        """QFT should produce correct phase pattern."""
        reset()
        # Create |2⟩ = |10⟩ (Q0=1, Q1=0 in LSB-first order, but we pass MSB first)
        pushQubit("Q0", [0, 1])  # MSB = 1
        pushQubit("Q1", [1, 0])  # LSB = 0
        # So this is |10⟩ binary = 2 decimal

        QFT(["Q0", "Q1"])
        state = get_state()

        # QFT|2⟩ = (1/2) Σ_k exp(2πi·2·k/4) |k⟩
        #        = (1/2) Σ_k exp(πik) |k⟩
        #        = (1/2) (|0⟩ - |1⟩ + |2⟩ - |3⟩)
        expected = np.array([1, -1, 1, -1]) / 2
        assert np.allclose(state, expected)
