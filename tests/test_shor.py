"""Tests for Shor's algorithm components."""

import numpy as np
import pytest

from qube import (
    reset, pushQubit, measureQubit, get_state,
    gcd, is_coprime,
    continued_fraction_expansion, convergents, extract_period,
    controlled_multiply_7_mod15, controlled_multiply_4_mod15,
    shor_factor_15,
)


class TestNumberTheory:
    """Tests for classical number theory utilities."""

    def test_gcd_basic(self):
        """GCD should return greatest common divisor."""
        assert gcd(15, 7) == 1
        assert gcd(15, 5) == 5
        assert gcd(15, 3) == 3
        assert gcd(12, 8) == 4

    def test_is_coprime(self):
        """is_coprime should correctly identify coprime pairs."""
        assert is_coprime(7, 15) == True
        assert is_coprime(5, 15) == False
        assert is_coprime(3, 15) == False
        assert is_coprime(11, 15) == True


class TestContinuedFractions:
    """Tests for continued fractions."""

    def test_expansion_of_quarter(self):
        """Continued fraction of 0.25 = 1/4."""
        cf = continued_fraction_expansion(0.25)
        assert cf == [0, 4]

    def test_expansion_of_half(self):
        """Continued fraction of 0.5 = 1/2."""
        cf = continued_fraction_expansion(0.5)
        assert cf == [0, 2]

    def test_convergents_basic(self):
        """Convergents should approximate the original value."""
        cf = continued_fraction_expansion(0.25)
        convs = convergents(cf)
        # Last convergent should be 1/4
        assert convs[-1] == (1, 4)


class TestPeriodExtraction:
    """Tests for period extraction."""

    def test_period_from_measurement_4(self):
        """Measurement 4 from 4-qubit register (phase 0.25) → period 4."""
        r = extract_period(4, 4, 15, 7)
        assert r == 4

    def test_period_from_measurement_8(self):
        """Measurement 8 (phase 0.5 = 2/4 reduced) should still give period 4."""
        # This tests the fix for reduced fractions
        r = extract_period(8, 4, 15, 7)
        assert r == 4, "Should handle reduced fraction 1/2 → period 4"

    def test_period_from_measurement_12(self):
        """Measurement 12 (phase 0.75 = 3/4) → period 4."""
        r = extract_period(12, 4, 15, 7)
        assert r == 4

    def test_period_from_measurement_0(self):
        """Measurement 0 gives no information."""
        r = extract_period(0, 4, 15, 7)
        assert r is None


class TestModularMultiplication:
    """Tests for controlled modular multiplication circuits."""

    def _classical_mod_mult(self, x, a, N):
        return (x * a) % N

    def test_multiply_7_mod15(self):
        """Controlled ×7 mod 15 should match classical computation."""
        test_values = [1, 2, 4, 7, 8, 11, 13, 14]

        for x in test_values:
            reset()

            # Encode x in work register (binary, LSB first)
            pushQubit("W0", [1 - (x & 1), x & 1])
            pushQubit("W1", [1 - ((x >> 1) & 1), (x >> 1) & 1])
            pushQubit("W2", [1 - ((x >> 2) & 1), (x >> 2) & 1])
            pushQubit("W3", [1 - ((x >> 3) & 1), (x >> 3) & 1])

            # Control qubit = |1⟩
            pushQubit("C", [0, 1])

            controlled_multiply_7_mod15("C", ["W0", "W1", "W2", "W3"])

            # Measure result
            result = 0
            for i, q in enumerate(["W0", "W1", "W2", "W3"]):
                bit = int(measureQubit(q))
                result += bit * (2 ** i)
            measureQubit("C")

            expected = self._classical_mod_mult(x, 7, 15)
            assert result == expected, f"{x} × 7 mod 15: got {result}, expected {expected}"

    def test_multiply_4_mod15(self):
        """Controlled ×4 mod 15 should match classical computation."""
        test_values = [1, 2, 4, 7, 8, 11, 13, 14]

        for x in test_values:
            reset()

            pushQubit("W0", [1 - (x & 1), x & 1])
            pushQubit("W1", [1 - ((x >> 1) & 1), (x >> 1) & 1])
            pushQubit("W2", [1 - ((x >> 2) & 1), (x >> 2) & 1])
            pushQubit("W3", [1 - ((x >> 3) & 1), (x >> 3) & 1])
            pushQubit("C", [0, 1])

            controlled_multiply_4_mod15("C", ["W0", "W1", "W2", "W3"])

            result = 0
            for i, q in enumerate(["W0", "W1", "W2", "W3"]):
                bit = int(measureQubit(q))
                result += bit * (2 ** i)
            measureQubit("C")

            expected = self._classical_mod_mult(x, 4, 15)
            assert result == expected, f"{x} × 4 mod 15: got {result}, expected {expected}"

    def test_multiply_no_op_when_control_zero(self):
        """Controlled multiplication should be identity when control=|0⟩."""
        test_values = [1, 2, 4, 7]

        for x in test_values:
            reset()

            pushQubit("W0", [1 - (x & 1), x & 1])
            pushQubit("W1", [1 - ((x >> 1) & 1), (x >> 1) & 1])
            pushQubit("W2", [1 - ((x >> 2) & 1), (x >> 2) & 1])
            pushQubit("W3", [1 - ((x >> 3) & 1), (x >> 3) & 1])
            pushQubit("C", [1, 0])  # Control = |0⟩

            controlled_multiply_7_mod15("C", ["W0", "W1", "W2", "W3"])

            result = 0
            for i, q in enumerate(["W0", "W1", "W2", "W3"]):
                bit = int(measureQubit(q))
                result += bit * (2 ** i)
            measureQubit("C")

            assert result == x, f"Control=0: {x} should remain {x}, got {result}"


class TestShorAlgorithm:
    """Tests for the complete Shor's algorithm."""

    def test_shor_returns_valid_factors_or_none(self):
        """Shor should return valid factors of 15 or None."""
        result = shor_factor_15()

        if result is not None:
            factor1, factor2 = result
            assert factor1 * factor2 == 15
            assert factor1 in [3, 5]
            assert factor2 in [3, 5]

    def test_shor_multiple_runs_finds_factors(self):
        """Multiple runs of Shor should find factors with high probability."""
        successes = 0
        n_trials = 20

        for _ in range(n_trials):
            result = shor_factor_15()
            if result is not None:
                factor1, factor2 = result
                assert factor1 * factor2 == 15
                successes += 1

        # Should succeed more often than not
        # With the fixes, measurement outcomes {4, 8, 12} all give r=4
        # Only measurement 0 fails, so success rate should be ~75%
        assert successes >= n_trials * 0.4, f"Success rate too low: {successes}/{n_trials}"


class TestModularExponentiationPeriod:
    """Tests verifying the period of 7^x mod 15."""

    def test_period_is_four(self):
        """7^4 mod 15 = 1, confirming period r=4."""
        assert pow(7, 4, 15) == 1

    def test_power_cycle(self):
        """7^x mod 15 cycles with period 4."""
        expected = [1, 7, 4, 13, 1, 7, 4, 13]
        actual = [pow(7, x, 15) for x in range(8)]
        assert actual == expected
