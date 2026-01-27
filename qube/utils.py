"""
Utility functions for quantum computing.

This module provides helper functions for:
- Quantum state comparison (accounting for global phase)
- Classical number theory (GCD, coprimality)
- Continued fractions for period extraction
"""

import numpy as np
from typing import List, Tuple, Optional


# =============================================================================
# Quantum state utilities
# =============================================================================

def allclose_up_to_global_phase(v, w, atol: float = 1e-9) -> bool:
    """
    Check if two quantum states are equal up to a global phase.

    This is the correct way to compare quantum states since global phase
    has no physical significance. Using np.abs() is incorrect for phase-
    sensitive gates like controlled-phase.

    Args:
        v: First quantum state (array-like)
        w: Second quantum state (array-like)
        atol: Absolute tolerance for comparison

    Returns:
        True if states are equal up to global phase
    """
    v = np.asarray(v).reshape(-1)
    w = np.asarray(w).reshape(-1)

    # Find a stable pivot amplitude in w
    idx = np.argmax(np.abs(w))
    if np.abs(w[idx]) < atol:
        # Both should be ~0 vectors; fallback to direct comparison
        return np.allclose(v, w, atol=atol)

    phase = v[idx] / w[idx]
    return np.allclose(v, phase * w, atol=atol)


def state_fidelity(v, w) -> float:
    """
    Compute the fidelity between two pure quantum states.

    Fidelity F = |⟨v|w⟩|² ranges from 0 (orthogonal) to 1 (identical).

    Args:
        v: First quantum state
        w: Second quantum state

    Returns:
        Fidelity value between 0 and 1
    """
    v = np.asarray(v).reshape(-1)
    w = np.asarray(w).reshape(-1)
    return np.abs(np.vdot(v, w)) ** 2


# =============================================================================
# Number theory utilities
# =============================================================================

def gcd(a: int, b: int) -> int:
    """
    Greatest common divisor using Euclidean algorithm.

    Args:
        a, b: Integers

    Returns:
        GCD of a and b
    """
    while b:
        a, b = b, a % b
    return a


def is_coprime(a: int, N: int) -> bool:
    """
    Check if a and N are coprime (share no common factors).

    Args:
        a, N: Integers to check

    Returns:
        True if gcd(a, N) == 1
    """
    return gcd(a, N) == 1


def mod_inverse(a: int, N: int) -> Optional[int]:
    """
    Compute modular inverse of a mod N using extended Euclidean algorithm.

    Args:
        a: Number to invert
        N: Modulus

    Returns:
        x such that (a * x) mod N == 1, or None if no inverse exists
    """
    if gcd(a, N) != 1:
        return None

    # Extended Euclidean algorithm
    old_r, r = a, N
    old_s, s = 1, 0

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s

    return old_s % N


# =============================================================================
# Continued fractions
# =============================================================================

def continued_fraction_expansion(x: float, max_terms: int = 20) -> List[int]:
    """
    Compute continued fraction expansion of x.

    Returns list of coefficients [a0, a1, a2, ...] where
    x ≈ a0 + 1/(a1 + 1/(a2 + ...))

    Args:
        x: Number to expand
        max_terms: Maximum number of terms

    Returns:
        List of continued fraction coefficients
    """
    coeffs = []
    for _ in range(max_terms):
        coeffs.append(int(x))
        frac = x - int(x)
        if frac < 1e-10:
            break
        x = 1 / frac
    return coeffs


def convergents(coeffs: List[int]) -> List[Tuple[int, int]]:
    """
    Compute convergents from continued fraction coefficients.

    Each convergent is a rational approximation to the original number.

    Args:
        coeffs: Continued fraction coefficients

    Returns:
        List of (numerator, denominator) pairs
    """
    convs = []
    h_prev, h_curr = 0, 1
    k_prev, k_curr = 1, 0

    for a in coeffs:
        h_new = a * h_curr + h_prev
        k_new = a * k_curr + k_prev
        convs.append((h_new, k_new))
        h_prev, h_curr = h_curr, h_new
        k_prev, k_curr = k_curr, k_new

    return convs


def extract_period(measurement: int, num_qubits: int, N: int, a: int) -> Optional[int]:
    """
    Extract period r from measurement result using continued fractions.

    This is used in Shor's algorithm to find the period of a^x mod N.
    The measurement gives an approximation to s/r for some integer s.

    When s and r share a common factor, the fraction s/r reduces, and
    continued fractions returns a divisor of r. We handle this by trying
    multiples of each candidate denominator.

    Args:
        measurement: Measured value from control register
        num_qubits: Number of qubits in control register
        N: Number to factor
        a: Base for modular exponentiation

    Returns:
        Candidate period r, or None if extraction failed
    """
    if measurement == 0:
        return None  # Measurement of 0 gives no information

    # The measured value is approximately s/r * 2^n
    # So measurement / 2^n ≈ s/r for some integer s
    phase = measurement / (2 ** num_qubits)

    # Use continued fractions to find r
    coeffs = continued_fraction_expansion(phase)
    convs = convergents(coeffs)

    # Look for convergent with denominator < N that gives valid period
    for num, denom in convs:
        if denom <= 0:
            continue

        # Try multiples of denom (handles reduced fractions like 1/2 when r=4)
        r = denom
        while r < N:
            if pow(a, r, N) == 1:
                return r
            r += denom

    return None


# =============================================================================
# Binary utilities
# =============================================================================

def int_to_bits(x: int, n: int) -> List[int]:
    """
    Convert integer to list of bits (LSB first).

    Args:
        x: Integer to convert
        n: Number of bits

    Returns:
        List of n bits, LSB first
    """
    return [(x >> i) & 1 for i in range(n)]


def bits_to_int(bits: List[int]) -> int:
    """
    Convert list of bits (LSB first) to integer.

    Args:
        bits: List of bits, LSB first

    Returns:
        Integer value
    """
    result = 0
    for i, bit in enumerate(bits):
        result += bit * (2 ** i)
    return result
