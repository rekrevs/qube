"""Tests for Draper QFT Adder."""

import pytest

from qube import reset
from qube.draper import (
    draper_add_constant,
    draper_add,
    draper_subtract,
    init_register,
    measure_register,
)


class TestDraperAddConstant:
    """Tests for adding classical constants to quantum registers."""

    @pytest.mark.parametrize(
        "initial_value,constant,expected",
        [
            (0, 3, 3),    # 0 + 3 = 3
            (5, 3, 8),    # 5 + 3 = 8
            (7, 1, 8),    # 7 + 1 = 8
            (5, 0, 5),    # 5 + 0 = 5 (adding zero)
        ],
        ids=["zero_plus_three", "five_plus_three", "seven_plus_one", "add_zero"],
    )
    def test_basic_addition(self, initial_value: int, constant: int, expected: int):
        """Test basic constant addition without overflow."""
        reset()
        qubits = init_register("a", initial_value, n_bits=4)
        draper_add_constant(qubits, constant)
        result = measure_register(qubits)
        assert result == expected, f"{initial_value} + {constant} = {result}, expected {expected}"

    @pytest.mark.parametrize(
        "initial_value,constant,expected",
        [
            (15, 1, 0),   # 15 + 1 = 16 mod 16 = 0
            (10, 7, 1),   # 10 + 7 = 17 mod 16 = 1
        ],
        ids=["overflow_15_plus_1", "overflow_10_plus_7"],
    )
    def test_overflow_wraps_around(self, initial_value: int, constant: int, expected: int):
        """Test that overflow wraps around correctly (mod 2^n)."""
        reset()
        qubits = init_register("a", initial_value, n_bits=4)
        draper_add_constant(qubits, constant)
        result = measure_register(qubits)
        assert result == expected, f"{initial_value} + {constant} mod 16 = {result}, expected {expected}"

    def test_different_register_sizes(self):
        """Test addition with different register sizes."""
        # 3-bit register: 5 + 2 = 7
        reset()
        qubits = init_register("a", 5, n_bits=3)
        draper_add_constant(qubits, 2)
        result = measure_register(qubits)
        assert result == 7

        # 5-bit register: 20 + 10 = 30
        reset()
        qubits = init_register("a", 20, n_bits=5)
        draper_add_constant(qubits, 10)
        result = measure_register(qubits)
        assert result == 30


class TestDraperAdd:
    """Tests for adding two quantum registers."""

    @pytest.mark.parametrize(
        "a_val,b_val,expected_a,expected_b",
        [
            (5, 3, 8, 3),    # 5 + 3 = 8, b unchanged
            (0, 7, 7, 7),    # 0 + 7 = 7, b unchanged
            (6, 6, 12, 6),   # 6 + 6 = 12, b unchanged
        ],
        ids=["five_plus_three", "zero_plus_seven", "six_plus_six"],
    )
    def test_basic_register_addition(
        self, a_val: int, b_val: int, expected_a: int, expected_b: int
    ):
        """Test basic addition of two quantum registers without overflow."""
        reset()
        a_qubits = init_register("a", a_val, n_bits=4)
        b_qubits = init_register("b", b_val, n_bits=4)

        draper_add(a_qubits, b_qubits)

        result_a = measure_register(a_qubits)
        result_b = measure_register(b_qubits)

        assert result_a == expected_a, f"a: {a_val} + {b_val} = {result_a}, expected {expected_a}"
        assert result_b == expected_b, f"b should be unchanged: got {result_b}, expected {expected_b}"

    @pytest.mark.parametrize(
        "a_val,b_val,expected_a,expected_b",
        [
            (15, 1, 0, 1),     # 15 + 1 = 16 mod 16 = 0
            (10, 10, 4, 10),   # 10 + 10 = 20 mod 16 = 4
        ],
        ids=["overflow_15_plus_1", "overflow_10_plus_10"],
    )
    def test_register_addition_overflow(
        self, a_val: int, b_val: int, expected_a: int, expected_b: int
    ):
        """Test that register addition overflow wraps around correctly."""
        reset()
        a_qubits = init_register("a", a_val, n_bits=4)
        b_qubits = init_register("b", b_val, n_bits=4)

        draper_add(a_qubits, b_qubits)

        result_a = measure_register(a_qubits)
        result_b = measure_register(b_qubits)

        assert result_a == expected_a, f"a: {a_val} + {b_val} mod 16 = {result_a}, expected {expected_a}"
        assert result_b == expected_b, f"b should be unchanged: got {result_b}, expected {expected_b}"

    def test_b_register_unchanged(self):
        """Verify that the b register is not modified by the addition."""
        reset()
        a_qubits = init_register("a", 3, n_bits=4)
        b_qubits = init_register("b", 5, n_bits=4)

        draper_add(a_qubits, b_qubits)

        result_b = measure_register(b_qubits)
        assert result_b == 5, "b register should remain unchanged after addition"


class TestDraperSubtract:
    """Tests for subtracting quantum registers."""

    @pytest.mark.parametrize(
        "a_val,b_val,expected",
        [
            (8, 3, 5),    # 8 - 3 = 5
            (5, 5, 0),    # 5 - 5 = 0
        ],
        ids=["eight_minus_three", "five_minus_five"],
    )
    def test_basic_subtraction(self, a_val: int, b_val: int, expected: int):
        """Test basic subtraction without underflow."""
        reset()
        a_qubits = init_register("a", a_val, n_bits=4)
        b_qubits = init_register("b", b_val, n_bits=4)

        draper_subtract(a_qubits, b_qubits)

        result = measure_register(a_qubits)
        assert result == expected, f"{a_val} - {b_val} = {result}, expected {expected}"

    @pytest.mark.parametrize(
        "a_val,b_val,expected",
        [
            (3, 5, 14),   # 3 - 5 = -2 mod 16 = 14
            (0, 1, 15),   # 0 - 1 = -1 mod 16 = 15
        ],
        ids=["wrap_3_minus_5", "wrap_0_minus_1"],
    )
    def test_subtraction_wraps_around(self, a_val: int, b_val: int, expected: int):
        """Test that underflow wraps around correctly (mod 2^n)."""
        reset()
        a_qubits = init_register("a", a_val, n_bits=4)
        b_qubits = init_register("b", b_val, n_bits=4)

        draper_subtract(a_qubits, b_qubits)

        result = measure_register(a_qubits)
        assert result == expected, f"{a_val} - {b_val} mod 16 = {result}, expected {expected}"

    def test_subtraction_is_inverse_of_addition(self):
        """Verify that subtracting reverses an addition."""
        reset()
        a_qubits = init_register("a", 7, n_bits=4)
        b_qubits = init_register("b", 4, n_bits=4)

        # Add: 7 + 4 = 11
        draper_add(a_qubits, b_qubits)
        result_after_add = measure_register(a_qubits)
        assert result_after_add == 11

        # Need to reinitialize for a clean test
        reset()
        a_qubits = init_register("a", 11, n_bits=4)
        b_qubits = init_register("b", 4, n_bits=4)

        # Subtract: 11 - 4 = 7
        draper_subtract(a_qubits, b_qubits)
        result_after_sub = measure_register(a_qubits)
        assert result_after_sub == 7


class TestEdgeCases:
    """Tests for edge cases in Draper adder."""

    def test_zero_register(self):
        """Test operations with zero register."""
        reset()
        a_qubits = init_register("a", 0, n_bits=4)
        b_qubits = init_register("b", 0, n_bits=4)

        draper_add(a_qubits, b_qubits)

        result_a = measure_register(a_qubits)
        result_b = measure_register(b_qubits)

        assert result_a == 0, "0 + 0 should equal 0"
        assert result_b == 0, "b should remain 0"

    def test_max_values(self):
        """Test with maximum values for the register size."""
        reset()
        # 4-bit register max = 15
        a_qubits = init_register("a", 15, n_bits=4)
        b_qubits = init_register("b", 15, n_bits=4)

        draper_add(a_qubits, b_qubits)

        result_a = measure_register(a_qubits)
        # 15 + 15 = 30 mod 16 = 14
        assert result_a == 14, f"15 + 15 mod 16 = {result_a}, expected 14"

    def test_single_bit_register(self):
        """Test with single-bit registers."""
        # 0 + 1 = 1
        reset()
        a_qubits = init_register("a", 0, n_bits=1)
        b_qubits = init_register("b", 1, n_bits=1)

        draper_add(a_qubits, b_qubits)

        result_a = measure_register(a_qubits)
        assert result_a == 1

        # 1 + 1 = 0 (overflow in single bit)
        reset()
        a_qubits = init_register("a", 1, n_bits=1)
        b_qubits = init_register("b", 1, n_bits=1)

        draper_add(a_qubits, b_qubits)

        result_a = measure_register(a_qubits)
        assert result_a == 0, "1 + 1 mod 2 = 0"
