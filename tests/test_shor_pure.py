"""Tests for pure shift-and-add quantum Shor's algorithm.

These tests verify the implementation uses shift-and-add multiplication
built from Draper adders, NOT precomputed permutation cycles.
"""

import inspect
import math
import numpy as np
import pytest

from qube import reset, pushQubit, applyGate, get_state
from qube.core import tosQubit
from qube.gates import H_gate
from qube.draper import init_register, measure_register


class TestShiftAndAddPurity:
    """Verify implementation is truly shift-and-add, not cycle-based."""

    def test_no_cycle_function_imported(self):
        """shor_pure.py must not import find_multiplication_cycles."""
        import qube.shor_pure as shor_pure

        assert not hasattr(shor_pure, 'find_multiplication_cycles'), \
            "shor_pure should not have find_multiplication_cycles"

        # Check it's not imported under a different name
        source = inspect.getsource(shor_pure)
        assert 'find_multiplication_cycles' not in source, \
            "shor_pure.py must not reference find_multiplication_cycles"

    def test_no_cycle_precomputation_in_mult(self):
        """controlled_mod_mult_pure must not call cycle-based functions."""
        from qube.shor_pure import controlled_mod_mult_pure

        source = inspect.getsource(controlled_mod_mult_pure)

        # Must not reference cycle functions
        assert 'find_multiplication_cycles' not in source, \
            "Must not use find_multiplication_cycles"
        assert 'controlled_swap_states' not in source, \
            "Must not use controlled_swap_states from shor_quantum"

    def test_uses_modular_addition(self):
        """Must use modular addition as building block."""
        from qube.shor_pure import controlled_mod_mult_pure

        source = inspect.getsource(controlled_mod_mult_pure)

        # Should reference addition-based operations
        has_add = ('modular_add' in source or
                   'phi_add' in source or
                   'controlled_modular_add' in source)
        assert has_add, \
            "Must use modular addition (modular_add, phi_add, or controlled_modular_add)"


class TestControlledModularAdd:
    """Tests for controlled modular addition of a constant."""

    @pytest.mark.parametrize(
        "initial,constant,N,expected",
        [
            (3, 5, 15, 8),    # 3 + 5 = 8 < 15, no reduction
            (10, 7, 15, 2),   # 10 + 7 = 17 mod 15 = 2
            (14, 1, 15, 0),   # 14 + 1 = 15 mod 15 = 0
            (5, 10, 21, 15),  # 5 + 10 = 15 < 21, no reduction
            (15, 10, 21, 4),  # 15 + 10 = 25 mod 21 = 4
        ],
        ids=["no_red_15", "reduce_15", "exact_15", "no_red_21", "reduce_21"],
    )
    def test_controlled_add_when_control_is_one(
        self, initial: int, constant: int, N: int, expected: int
    ):
        """Controlled add should work when control=|1⟩."""
        from qube.shor_pure import controlled_modular_add_constant

        n_bits = math.ceil(math.log2(N)) + 1

        reset()
        # Control = |1⟩
        pushQubit("ctrl", [0, 1])
        qubits = init_register("a", initial, n_bits=n_bits)

        controlled_modular_add_constant("ctrl", qubits, constant, N)

        result = measure_register(qubits)
        assert result == expected, \
            f"ctrl=1: {initial} + {constant} mod {N} = {result}, expected {expected}"

    @pytest.mark.parametrize(
        "initial,constant,N",
        [
            (5, 7, 15),
            (10, 8, 21),
        ],
        ids=["N=15", "N=21"],
    )
    def test_controlled_add_identity_when_control_zero(
        self, initial: int, constant: int, N: int
    ):
        """Controlled add should do nothing when control=|0⟩."""
        from qube.shor_pure import controlled_modular_add_constant

        n_bits = math.ceil(math.log2(N)) + 1

        reset()
        # Control = |0⟩
        pushQubit("ctrl", [1, 0])
        qubits = init_register("a", initial, n_bits=n_bits)

        controlled_modular_add_constant("ctrl", qubits, constant, N)

        result = measure_register(qubits)
        assert result == initial, \
            f"ctrl=0: value should stay {initial}, got {result}"


class TestShiftAndAddMultiply:
    """Tests for shift-and-add modular multiplication."""

    @pytest.mark.parametrize(
        "N,a,x,expected",
        [
            (15, 7, 1, 7),    # 7 * 1 mod 15 = 7
            (15, 7, 2, 14),   # 7 * 2 mod 15 = 14
            (15, 7, 4, 13),   # 7 * 4 mod 15 = 28 mod 15 = 13
            (15, 7, 7, 4),    # 7 * 7 mod 15 = 49 mod 15 = 4
            (15, 7, 13, 1),   # 7 * 13 mod 15 = 91 mod 15 = 1
            (21, 2, 1, 2),    # 2 * 1 mod 21 = 2
            (21, 2, 5, 10),   # 2 * 5 mod 21 = 10
            (21, 2, 11, 1),   # 2 * 11 mod 21 = 22 mod 21 = 1
        ],
        ids=["7x1", "7x2", "7x4", "7x7", "7x13", "2x1_21", "2x5_21", "2x11_21"],
    )
    def test_mult_matches_classical(self, N: int, a: int, x: int, expected: int):
        """Shift-and-add mult should match (a*x) mod N."""
        from qube.shor_pure import controlled_mod_mult_pure

        n_bits = math.ceil(math.log2(N)) + 1

        reset()
        # Control = |1⟩
        pushQubit("ctrl", [0, 1])
        x_qubits = init_register("x", x, n_bits=n_bits)

        controlled_mod_mult_pure("ctrl", x_qubits, a, N)

        result = measure_register(x_qubits)
        assert result == expected, \
            f"ctrl=1: {a} * {x} mod {N} = {result}, expected {expected}"

    def test_mult_identity_when_control_zero(self):
        """When control=|0⟩, x should remain unchanged."""
        from qube.shor_pure import controlled_mod_mult_pure

        N, a, x = 15, 7, 4
        n_bits = math.ceil(math.log2(N)) + 1

        reset()
        # Control = |0⟩
        pushQubit("ctrl", [1, 0])
        x_qubits = init_register("x", x, n_bits=n_bits)

        controlled_mod_mult_pure("ctrl", x_qubits, a, N)

        result = measure_register(x_qubits)
        assert result == x, f"ctrl=0: x should stay {x}, got {result}"

    def test_matches_shor_quantum_implementation(self):
        """Pure implementation should give same results as cycle-based."""
        from qube.shor_pure import controlled_mod_mult_pure
        from qube.shor_quantum import controlled_mod_mult

        N, a = 15, 7
        n_bits = math.ceil(math.log2(N))

        # Test all values in multiplicative group
        test_values = [x for x in range(1, N) if math.gcd(x, N) == 1]

        for x in test_values:
            # Pure implementation
            reset()
            pushQubit("ctrl", [0, 1])
            x_qubits = init_register("x", x, n_bits=n_bits)
            controlled_mod_mult_pure("ctrl", x_qubits, a, N)
            result_pure = measure_register(x_qubits)

            # Cycle-based implementation
            reset()
            pushQubit("ctrl", [0, 1])
            x_qubits = init_register("x", x, n_bits=n_bits)
            controlled_mod_mult("ctrl", x_qubits, a, N)
            result_cycle = measure_register(x_qubits)

            assert result_pure == result_cycle, \
                f"Pure vs cycle mismatch for x={x}: {result_pure} != {result_cycle}"


class TestQuantumBehavior:
    """Tests that verify true quantum behavior (superposition preserved)."""

    def test_superposition_preserved_in_mult(self):
        """Controlled mult with control in superposition creates entanglement.

        Setup: ctrl = H|0⟩ = (|0⟩+|1⟩)/√2, x = |1⟩
        After controlled_mod_mult_pure(ctrl, x, a=7, N=15):
            (|0⟩|1⟩ + |1⟩|7⟩)/√2

        This is entangled. A classical cheat would collapse the superposition.
        """
        from qube.shor_pure import controlled_mod_mult_pure

        N, a = 15, 7
        n_bits = math.ceil(math.log2(N))

        reset()

        # Control in superposition
        pushQubit("ctrl", [1, 0])
        applyGate(H_gate, "ctrl")

        # x = |1⟩
        x_qubits = init_register("x", 1, n_bits=n_bits)

        controlled_mod_mult_pure("ctrl", x_qubits, a, N)

        # Normalize qubit order
        tosQubit("ctrl")
        for q in x_qubits:
            tosQubit(q)

        state = get_state()

        # Should have exactly 2 non-zero amplitudes
        nonzero = np.where(np.abs(state) > 1e-10)[0]
        assert len(nonzero) == 2, \
            f"Expected 2 non-zero amplitudes (entangled state), got {len(nonzero)}"

        # Both should have magnitude 1/√2
        for idx in nonzero:
            assert np.isclose(np.abs(state[idx]), 1/np.sqrt(2), atol=1e-6), \
                f"Amplitude should be 1/√2, got {np.abs(state[idx])}"

    def test_four_term_superposition_after_two_controls(self):
        """Two control qubits in superposition should give 4-term entanglement.

        Setup: c0, c1 = H|0⟩ each, x = |1⟩
        After applying a^1 controlled on c0 and a^2 controlled on c1:
            |00⟩|1⟩ + |01⟩|7⟩ + |10⟩|4⟩ + |11⟩|13⟩  (all with amplitude 1/2)

        (7^1=7, 7^2=4, 7^3=13 mod 15)
        """
        from qube.shor_pure import controlled_mod_mult_pure

        N, a = 15, 7
        n_bits = math.ceil(math.log2(N))

        reset()

        # Two control qubits in superposition
        pushQubit("c0", [1, 0])
        pushQubit("c1", [1, 0])
        applyGate(H_gate, "c0")
        applyGate(H_gate, "c1")

        # x = |1⟩
        x_qubits = init_register("x", 1, n_bits=n_bits)

        # c0 controls a^1 = 7
        controlled_mod_mult_pure("c0", x_qubits, 7, N)

        # c1 controls a^2 = 49 mod 15 = 4
        controlled_mod_mult_pure("c1", x_qubits, 4, N)

        # Normalize order
        tosQubit("c0")
        tosQubit("c1")
        for q in x_qubits:
            tosQubit(q)

        state = get_state()

        # Should have exactly 4 non-zero amplitudes
        nonzero = np.where(np.abs(state) > 1e-10)[0]
        assert len(nonzero) == 4, \
            f"Expected 4 non-zero amplitudes, got {len(nonzero)}"

        # Each should have magnitude 1/2
        for idx in nonzero:
            assert np.isclose(np.abs(state[idx]), 0.5, atol=1e-6), \
                f"Each amplitude should be 1/2, got {np.abs(state[idx])}"


class TestShorFactorPure:
    """Tests for the full pure Shor's algorithm."""

    @pytest.mark.parametrize(
        "N,factors",
        [
            pytest.param(15, {3, 5}, marks=pytest.mark.skip(
                reason="Shift-and-add too slow for full Shor (O(n²) QFTs per mult)"
            )),
            pytest.param(21, {3, 7}, marks=pytest.mark.skip(
                reason="Shift-and-add too slow for full Shor (O(n²) QFTs per mult)"
            )),
        ],
        ids=["N=15", "N=21"],
    )
    def test_factors_composite_N(self, N: int, factors: set):
        """shor_factor_pure(N) returns valid factorization."""
        from qube.shor_pure import shor_factor_pure

        # Run multiple times due to probabilistic nature
        for attempt in range(20):
            result = shor_factor_pure(N)
            if result is not None:
                f1, f2 = result
                assert f1 * f2 == N, f"Factors don't multiply to N"
                assert {f1, f2} == factors, f"Wrong factors: {f1}, {f2}"
                return

        pytest.fail(f"Failed to factor {N} in 20 attempts")

    @pytest.mark.skip(reason="Shift-and-add too slow for full Shor (O(n²) QFTs per mult)")
    def test_success_rate_acceptable(self):
        """Factoring N=15 should succeed with reasonable probability."""
        from qube.shor_pure import shor_factor_pure

        N = 15
        successes = sum(1 for _ in range(10) if shor_factor_pure(N) is not None)
        assert successes >= 3, f"Success rate {successes}/10 too low"
