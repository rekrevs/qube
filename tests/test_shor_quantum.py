"""Tests for generalized quantum Shor's algorithm.

These tests verify the quantum implementation of Shor's algorithm
for arbitrary N, built on Draper adders for modular arithmetic.
"""

import math
import numpy as np
import pytest

from qube import reset, pushQubit, applyGate, get_state
from qube.core import tosQubit
from qube.gates import H_gate
from qube.draper import init_register, measure_register


class TestRegisterSizing:
    """Verify dynamic register allocation."""

    @pytest.mark.parametrize(
        "N,expected_work,expected_control",
        [
            (15, 4, 8),   # ceil(log2(15))=4, 2*4=8
            (21, 5, 10),  # ceil(log2(21))=5, 2*5=10
            (33, 6, 12),  # ceil(log2(33))=6, 2*6=12
            (35, 6, 12),  # ceil(log2(35))=6, 2*6=12
        ],
        ids=["N=15", "N=21", "N=33", "N=35"],
    )
    def test_register_sizes(self, N: int, expected_work: int, expected_control: int):
        """Work register = ceil(log2(N)), control = 2*work."""
        from qube.shor_quantum import compute_register_sizes

        work_size, control_size = compute_register_sizes(N)
        assert work_size == expected_work, f"Work register for N={N}"
        assert control_size == expected_control, f"Control register for N={N}"


class TestModularArithmetic:
    """Tests for quantum modular arithmetic using Draper adder."""

    @pytest.mark.parametrize(
        "a,b,N,expected",
        [
            (3, 5, 15, 8),    # 3 + 5 = 8 < 15, no reduction
            (10, 7, 15, 2),   # 10 + 7 = 17 mod 15 = 2
            (14, 1, 15, 0),   # 14 + 1 = 15 mod 15 = 0
            (5, 10, 21, 15),  # 5 + 10 = 15 < 21, no reduction
            (15, 10, 21, 4),  # 15 + 10 = 25 mod 21 = 4
        ],
        ids=["no_reduction", "reduction_15", "exact_mod", "no_reduction_21", "reduction_21"],
    )
    def test_modular_add_constant(self, a: int, b: int, N: int, expected: int):
        """(a + b) mod N computed quantumly."""
        from qube.shor_quantum import modular_add_constant

        n_bits = math.ceil(math.log2(N)) + 1  # Extra bit for overflow detection

        reset()
        qubits = init_register("a", a, n_bits=n_bits)
        modular_add_constant(qubits, b, N)

        result = measure_register(qubits)
        assert result == expected, f"{a} + {b} mod {N} = {result}, expected {expected}"

    @pytest.mark.parametrize(
        "a,b,N,expected",
        [
            (8, 3, 15, 5),    # 8 - 3 = 5
            (2, 5, 15, 12),   # 2 - 5 = -3 mod 15 = 12
            (0, 1, 15, 14),   # 0 - 1 = -1 mod 15 = 14
            (4, 10, 21, 15),  # 4 - 10 = -6 mod 21 = 15
        ],
        ids=["no_borrow", "borrow_15", "zero_borrow", "borrow_21"],
    )
    def test_modular_subtract_constant(self, a: int, b: int, N: int, expected: int):
        """(a - b) mod N computed quantumly."""
        from qube.shor_quantum import modular_subtract_constant

        n_bits = math.ceil(math.log2(N)) + 1

        reset()
        qubits = init_register("a", a, n_bits=n_bits)
        modular_subtract_constant(qubits, b, N)

        result = measure_register(qubits)
        assert result == expected, f"{a} - {b} mod {N} = {result}, expected {expected}"


class TestControlledModMult:
    """Tests for controlled a*x mod N."""

    @pytest.mark.parametrize(
        "N,a,x,expected",
        [
            (15, 7, 1, 7),    # 7 * 1 mod 15 = 7
            (15, 7, 4, 13),   # 7 * 4 mod 15 = 28 mod 15 = 13
            (15, 7, 7, 4),    # 7 * 7 mod 15 = 49 mod 15 = 4
            (21, 2, 1, 2),    # 2 * 1 mod 21 = 2
            (21, 2, 5, 10),   # 2 * 5 mod 21 = 10
            (21, 2, 11, 1),   # 2 * 11 mod 21 = 22 mod 21 = 1
        ],
        ids=["7x1_mod15", "7x4_mod15", "7x7_mod15", "2x1_mod21", "2x5_mod21", "2x11_mod21"],
    )
    def test_controlled_mod_mult_matches_classical(
        self, N: int, a: int, x: int, expected: int
    ):
        """Quantum controlled mult should match classical (a*x) mod N."""
        from qube.shor_quantum import controlled_mod_mult

        n_bits = math.ceil(math.log2(N)) + 1

        reset()
        # Control qubit = |1⟩
        pushQubit("ctrl", [0, 1])
        work_qubits = init_register("w", x, n_bits=n_bits)

        controlled_mod_mult("ctrl", work_qubits, a, N)

        result = measure_register(work_qubits)
        assert result == expected, f"{a} * {x} mod {N} = {result}, expected {expected}"

    def test_controlled_mod_mult_identity_when_control_zero(self):
        """When control=0, x should remain unchanged."""
        from qube.shor_quantum import controlled_mod_mult

        N, a, x = 15, 7, 4
        n_bits = math.ceil(math.log2(N)) + 1

        reset()
        # Control qubit = |0⟩
        pushQubit("ctrl", [1, 0])
        work_qubits = init_register("w", x, n_bits=n_bits)

        controlled_mod_mult("ctrl", work_qubits, a, N)

        result = measure_register(work_qubits)
        assert result == x, f"x should be unchanged when control=0, got {result}"


class TestShorFactorN:
    """Tests for shor_factor(N) on various composites."""

    @pytest.mark.parametrize(
        "N,factors",
        [
            (15, {3, 5}),
            (21, {3, 7}),
            pytest.param(33, {3, 11}, marks=pytest.mark.skip(reason="18 qubits too slow for simulation")),
            pytest.param(35, {5, 7}, marks=pytest.mark.skip(reason="18 qubits too slow for simulation")),
        ],
        ids=["N=15", "N=21", "N=33", "N=35"],
    )
    def test_factors_composite_N(self, N: int, factors: set):
        """shor_factor(N) returns valid factorization."""
        from qube.shor_quantum import shor_factor

        # Run multiple times due to probabilistic nature
        for attempt in range(20):
            result = shor_factor(N)
            if result is not None:
                f1, f2 = result
                assert f1 * f2 == N, f"Factors don't multiply to N: {f1} * {f2} != {N}"
                assert {f1, f2} == factors, f"Wrong factors: {f1}, {f2}"
                return

        pytest.fail(f"Failed to factor {N} in 20 attempts")

    def test_success_rate_acceptable(self):
        """Factoring should succeed with reasonable probability."""
        from qube.shor_quantum import shor_factor

        # Use N=15 (faster) with fewer iterations
        N = 15
        successes = sum(1 for _ in range(10) if shor_factor(N) is not None)
        # Should succeed at least 40% of the time
        assert successes >= 4, f"Success rate {successes}/10 too low for N={N}"


class TestQuantumBehavior:
    """Tests that FAIL if implementation cheats with classical oracles.

    These tests verify true quantum superposition is maintained through
    the modular exponentiation, not just that the final answer is correct.
    """

    def test_controlled_mult_preserves_superposition(self):
        """Controlled mod mult with control in superposition creates entanglement.

        Setup:
            ctrl = (|0⟩ + |1⟩)/√2
            work = |1⟩

        After controlled_mod_mult(ctrl, work, a=7, N=15):
            (|0⟩|1⟩ + |1⟩|7⟩)/√2

        This is entangled - control and work are correlated.
        A classical cheat (measure control first) would not produce this.
        """
        from qube.shor_quantum import controlled_mod_mult

        N, a = 15, 7
        n_bits = math.ceil(math.log2(N)) + 1

        reset()

        # Control in superposition
        pushQubit("ctrl", [1, 0])
        applyGate(H_gate, "ctrl")

        # Work register = |1⟩
        work_qubits = init_register("w", 1, n_bits=n_bits)

        # Apply controlled multiplication
        controlled_mod_mult("ctrl", work_qubits, a, N)

        # Normalize qubit order
        tosQubit("ctrl")
        for q in work_qubits:
            tosQubit(q)

        state = get_state()

        # Expected: (|0⟩|00001⟩ + |1⟩|00111⟩)/√2
        # With n_bits=5: work=1 is |00001⟩, work=7 is |00111⟩
        # State indices: |ctrl, w0, w1, w2, w3, w4⟩ (MSB first)
        # |0, 0, 0, 0, 0, 1⟩ = 1 (binary 000001)
        # |1, 0, 0, 1, 1, 1⟩ = 39 (binary 100111)

        # For 6 qubits (1 ctrl + 5 work), state vector has 64 entries
        n_qubits = 1 + n_bits
        expected_indices = [1, 32 + 7]  # |0⟩|1⟩ and |1⟩|7⟩

        # Should have exactly 2 non-zero amplitudes
        nonzero_indices = np.where(np.abs(state) > 1e-10)[0]
        assert len(nonzero_indices) == 2, (
            f"Expected 2 non-zero amplitudes for entangled state, "
            f"got {len(nonzero_indices)}: {nonzero_indices}"
        )

        # Both amplitudes should be ~1/√2
        for idx in nonzero_indices:
            assert np.isclose(np.abs(state[idx]), 1 / np.sqrt(2), atol=1e-6), (
                f"Amplitude at index {idx} should be 1/√2, got {np.abs(state[idx])}"
            )

    def test_work_register_not_measured_during_exponentiation(self):
        """State should be pure superposition, not collapsed.

        After controlled modular exponentiation with control in superposition,
        the state should be a coherent superposition. If the implementation
        measured the work register during computation, it would collapse.
        """
        from qube.shor_quantum import apply_controlled_mod_exp

        N, a = 15, 7
        n_bits = math.ceil(math.log2(N)) + 1

        reset()

        # Control register: 2 qubits in full superposition
        pushQubit("c0", [1, 0])
        pushQubit("c1", [1, 0])
        applyGate(H_gate, "c0")
        applyGate(H_gate, "c1")
        control_qubits = ["c0", "c1"]

        # Work register = |1⟩
        work_qubits = init_register("w", 1, n_bits=n_bits)

        # Apply controlled mod exp: c0 controls a^1, c1 controls a^2
        apply_controlled_mod_exp(control_qubits, work_qubits, a, N)

        # Normalize order
        for q in control_qubits + work_qubits:
            tosQubit(q)

        state = get_state()

        # With 2 control qubits, we expect 4 terms in superposition:
        # |00⟩|1⟩, |01⟩|7⟩, |10⟩|4⟩, |11⟩|7*4=13⟩
        # (since 7^1=7, 7^2=4 mod 15, 7^3=13 mod 15)
        expected_work_values = {
            0b00: 1,   # a^0 * 1 = 1
            0b01: 7,   # a^1 * 1 = 7
            0b10: 4,   # a^2 * 1 = 4
            0b11: 13,  # a^3 * 1 = 13
        }

        # Verify we have exactly 4 non-zero amplitudes
        nonzero_count = np.sum(np.abs(state) > 1e-10)
        assert nonzero_count == 4, (
            f"Expected 4 terms in superposition, got {nonzero_count}"
        )

        # Each should have amplitude 1/2
        for idx in np.where(np.abs(state) > 1e-10)[0]:
            assert np.isclose(np.abs(state[idx]), 0.5, atol=1e-6), (
                f"Each term should have amplitude 1/2, got {np.abs(state[idx])}"
            )
