"""Tests for quantum gates and core operations."""

import numpy as np
import pytest

from qube import (
    reset, pushQubit, applyGate, measureQubit, get_state,
    H_gate, X_gate, Y_gate, Z_gate, CNOT_gate, SWAP_gate, TOFF_gate,
    P_gate, CP_gate, CP_decomposed,
    allclose_up_to_global_phase,
)


class TestSingleQubitGates:
    """Tests for single-qubit gates."""

    def test_x_gate_flips_zero_to_one(self):
        """X gate should flip |0⟩ to |1⟩."""
        reset()
        pushQubit("q", [1, 0])  # |0⟩
        applyGate(X_gate, "q")
        state = get_state()
        assert np.allclose(state, [0, 1])

    def test_x_gate_flips_one_to_zero(self):
        """X gate should flip |1⟩ to |0⟩."""
        reset()
        pushQubit("q", [0, 1])  # |1⟩
        applyGate(X_gate, "q")
        state = get_state()
        assert np.allclose(state, [1, 0])

    def test_h_gate_creates_superposition(self):
        """H gate on |0⟩ should create equal superposition."""
        reset()
        pushQubit("q", [1, 0])
        applyGate(H_gate, "q")
        state = get_state()
        expected = np.array([1, 1]) / np.sqrt(2)
        assert np.allclose(state, expected)

    def test_h_gate_on_one(self):
        """H gate on |1⟩ should create |−⟩."""
        reset()
        pushQubit("q", [0, 1])
        applyGate(H_gate, "q")
        state = get_state()
        expected = np.array([1, -1]) / np.sqrt(2)
        assert np.allclose(state, expected)

    def test_h_gate_is_self_inverse(self):
        """H² = I."""
        reset()
        pushQubit("q", [0.6, 0.8])
        original = get_state().copy()
        applyGate(H_gate, "q")
        applyGate(H_gate, "q")
        state = get_state()
        assert np.allclose(state, original)

    def test_z_gate_flips_phase(self):
        """Z gate should flip phase of |1⟩."""
        reset()
        pushQubit("q", [1, 1])  # Will be normalized
        applyGate(Z_gate, "q")
        state = get_state()
        expected = np.array([1, -1]) / np.sqrt(2)
        assert np.allclose(state, expected)


class TestTwoQubitGates:
    """Tests for two-qubit gates."""

    def test_cnot_controlled_flip(self):
        """CNOT should flip target when control is |1⟩."""
        reset()
        pushQubit("control", [0, 1])  # |1⟩
        pushQubit("target", [1, 0])   # |0⟩
        applyGate(CNOT_gate, "control", "target")
        # State should be |11⟩
        state = get_state()
        expected = np.array([0, 0, 0, 1])
        assert np.allclose(state, expected)

    def test_cnot_no_flip_when_control_zero(self):
        """CNOT should not flip target when control is |0⟩."""
        reset()
        pushQubit("control", [1, 0])  # |0⟩
        pushQubit("target", [1, 0])   # |0⟩
        applyGate(CNOT_gate, "control", "target")
        # State should be |00⟩
        state = get_state()
        expected = np.array([1, 0, 0, 0])
        assert np.allclose(state, expected)

    def test_swap_gate(self):
        """SWAP should exchange two qubits."""
        reset()
        pushQubit("a", [1, 0])  # |0⟩
        pushQubit("b", [0, 1])  # |1⟩
        applyGate(SWAP_gate, "a", "b")
        # State should now be |10⟩ (a=1, b=0)
        state = get_state()
        expected = np.array([0, 0, 1, 0])
        assert np.allclose(state, expected)


class TestThreeQubitGates:
    """Tests for three-qubit gates."""

    def test_toffoli_truth_table(self):
        """Toffoli gate should implement AND."""
        truth_table = [
            ([1, 0], [1, 0], [1, 0], [1, 0]),  # 000 -> 000
            ([1, 0], [1, 0], [0, 1], [0, 1]),  # 001 -> 001
            ([1, 0], [0, 1], [1, 0], [1, 0]),  # 010 -> 010
            ([1, 0], [0, 1], [0, 1], [0, 1]),  # 011 -> 011
            ([0, 1], [1, 0], [1, 0], [1, 0]),  # 100 -> 100
            ([0, 1], [1, 0], [0, 1], [0, 1]),  # 101 -> 101
            ([0, 1], [0, 1], [1, 0], [0, 1]),  # 110 -> 111 (flip!)
            ([0, 1], [0, 1], [0, 1], [1, 0]),  # 111 -> 110 (flip!)
        ]

        for c1, c2, t_in, t_expected in truth_table:
            reset()
            pushQubit("c1", c1)
            pushQubit("c2", c2)
            pushQubit("t", t_in)
            applyGate(TOFF_gate, "c1", "c2", "t")
            result = measureQubit("t")
            expected = "1" if t_expected[1] > 0.5 else "0"
            assert result == expected


class TestControlledPhase:
    """Tests for controlled-phase gates."""

    def test_cp_decomposition_matches_matrix(self):
        """CP_decomposed should match CP_gate for all basis states."""
        from qube.core import tosQubit

        test_angles = [np.pi / 2, np.pi / 4, np.pi / 8, np.pi / 16]

        for theta in test_angles:
            for b1 in [0, 1]:
                for b2 in [0, 1]:
                    # Matrix version
                    reset()
                    pushQubit("C", [1 - b1, b1])
                    pushQubit("T", [1 - b2, b2])
                    applyGate(CP_gate(theta), "C", "T")
                    # Normalize qubit order to [C, T] for comparison
                    tosQubit("C")
                    tosQubit("T")
                    state_matrix = get_state().copy()

                    # Decomposed version
                    reset()
                    pushQubit("C", [1 - b1, b1])
                    pushQubit("T", [1 - b2, b2])
                    CP_decomposed(theta, "C", "T")
                    # Normalize qubit order to [C, T] for comparison
                    tosQubit("C")
                    tosQubit("T")
                    state_decomp = get_state().copy()

                    # Compare up to global phase (not abs - that destroys phase info!)
                    assert allclose_up_to_global_phase(state_matrix, state_decomp), \
                        f"Mismatch at theta={theta}, input=|{b1}{b2}>"


class TestMeasurement:
    """Tests for measurement."""

    def test_measurement_collapses_state(self):
        """Measurement should collapse superposition to basis state."""
        reset()
        pushQubit("q", [1, 0])
        applyGate(H_gate, "q")  # Now in superposition

        result = measureQubit("q")
        assert result in ["0", "1"]

    def test_measurement_statistics(self):
        """Measurement statistics should match probabilities."""
        counts = {"0": 0, "1": 0}
        n_trials = 1000

        for _ in range(n_trials):
            reset()
            pushQubit("q", [1, 0])
            applyGate(H_gate, "q")
            result = measureQubit("q")
            counts[result] += 1

        # Should be approximately 50/50, allow 10% margin
        assert 0.4 < counts["0"] / n_trials < 0.6
        assert 0.4 < counts["1"] / n_trials < 0.6
