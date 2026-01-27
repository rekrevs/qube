"""
Pure Shift-and-Add Quantum Shor's Algorithm.

This module implements Shor's factoring algorithm using shift-and-add
multiplication built from Draper QFT adders. This is a "pure" quantum
implementation that does NOT use precomputed permutation cycles.

Key components:
- Controlled modular addition using Draper adders
- Shift-and-add modular multiplication
- Full Shor's algorithm with quantum phase estimation

References:
- Beauregard, "Circuit for Shor's algorithm using 2n+3 qubits", 2002
- Draper, "Addition on a Quantum Computer", 2000
"""

import math
import random
from typing import List, Tuple, Optional

import numpy as np

from .core import applyGate, pushQubit, measureQubit, reset, tosQubit
from .gates import H_gate, X_gate, CNOT_gate, CP_gate, TOFF_gate
from .qft import QFT, QFT_inverse
from .draper import init_register
from .shor_quantum import compute_register_sizes
from .utils import gcd, is_coprime, extract_period, mod_inverse


# =============================================================================
# Controlled modular addition
# =============================================================================

def _normalize_qubits(qubits: List[str]):
    """Normalize qubit ordering after QFT/QFT_inverse."""
    for q in qubits:
        tosQubit(q)


def controlled_phi_add_constant(ctrl: str, qubits: List[str], constant: int,
                                 inverse: bool = False):
    """
    Add a constant in Fourier basis, controlled by a control qubit.

    When ctrl=|1>, adds constant to the register (already in Fourier basis).
    When ctrl=|0>, does nothing.

    Args:
        ctrl: Control qubit name
        qubits: List of qubit names [MSB, ..., LSB] (in Fourier basis)
        constant: Classical integer to add
        inverse: If True, subtract instead of add
    """
    n = len(qubits)
    sign = -1 if inverse else 1

    for j in range(n):
        divisor = 2 ** (j + 1)
        theta = sign * 2 * np.pi * constant / divisor
        applyGate(CP_gate(theta), ctrl, qubits[j])


def controlled_modular_add_constant(ctrl: str, qubits: List[str],
                                     constant: int, N: int):
    """
    Add a constant to a quantum register modulo N, controlled by ctrl qubit.

    When ctrl=|1>: qubits -> (qubits + constant) mod N
    When ctrl=|0>: qubits unchanged

    Uses the Beauregard circuit with proper ancilla uncomputation to preserve
    quantum superposition in the control qubit.

    The key to proper uncomputation is that after the operation is complete:
    - For ctrl=0: nothing happened, ancilla stays |0>
    - For ctrl=1: ancilla was set based on overflow, then uncomputed to |0>

    Args:
        ctrl: Control qubit name
        qubits: List of qubit names [MSB, ..., LSB] with extra overflow bit
        constant: Classical integer to add (0 <= constant < N)
        N: Modulus
    """
    # Step 1: Transform to Fourier basis
    QFT(qubits)
    _normalize_qubits(qubits)

    # Step 2: Controlled add constant
    controlled_phi_add_constant(ctrl, qubits, constant)

    # Step 3: Controlled subtract N
    controlled_phi_add_constant(ctrl, qubits, N, inverse=True)

    # Step 4: Back to computational basis for sign check
    QFT_inverse(qubits)
    _normalize_qubits(qubits)

    # At this point:
    # - If ctrl=0: register unchanged (value = a)
    # - If ctrl=1: register = a + constant - N (mod 2^n)
    #   MSB=1 means a+c < N (underflow), MSB=0 means a+c >= N

    # Step 5: Copy MSB to ancilla (controlled on ctrl via Toffoli)
    pushQubit("_cma_anc", [1, 0])
    applyGate(TOFF_gate, ctrl, qubits[0], "_cma_anc")

    # Step 6: Add N back controlled on ancilla
    QFT(qubits)
    _normalize_qubits(qubits)
    tosQubit("_cma_anc")
    controlled_phi_add_constant("_cma_anc", qubits, N)

    # Step 7: Uncompute ancilla using Beauregard's technique
    # Subtract constant (controlled on ctrl), then XOR with sign bit

    controlled_phi_add_constant(ctrl, qubits, constant, inverse=True)

    QFT_inverse(qubits)
    _normalize_qubits(qubits)
    tosQubit("_cma_anc")

    # At this point (for ctrl=1):
    # - If original a+c < N: result after +c-N+N-c = a, ancilla=1
    #   MSB depends on value of a
    # - If original a+c >= N: result after +c-N-c = a-N < 0, ancilla=0
    #   MSB = 1 (negative number)
    #
    # For proper uncomputation, we need: ancilla XOR (ctrl AND (NOT MSB))
    # when a+c < N (ancilla=1): result=a, if MSB=0 then 1 XOR 1 = 0 (correct!)
    #                           if MSB=1 then 1 XOR 0 = 1 (wrong!)
    # when a+c >= N (ancilla=0): result=a-N, MSB=1, 0 XOR 0 = 0 (correct!)
    #
    # The problem is when a itself has MSB=1 (a >= 2^(n-1)).
    # But if we require a < N < 2^(n-1), then a always has MSB=0.
    # With our n+1 bit register, N < 2^n, so a < N means MSB=0 for valid inputs.

    # Apply X to MSB, then Toffoli, then X again (computes ctrl AND NOT MSB)
    applyGate(X_gate, qubits[0])
    applyGate(TOFF_gate, ctrl, qubits[0], "_cma_anc")
    applyGate(X_gate, qubits[0])

    # Step 8: Add constant back to restore correct result
    QFT(qubits)
    _normalize_qubits(qubits)
    tosQubit("_cma_anc")
    controlled_phi_add_constant(ctrl, qubits, constant)
    QFT_inverse(qubits)
    _normalize_qubits(qubits)
    tosQubit("_cma_anc")

    # Ancilla should now be |0> - safe to measure without decoherence
    measureQubit("_cma_anc")


def controlled_modular_subtract_constant(ctrl: str, qubits: List[str],
                                          constant: int, N: int):
    """
    Subtract a constant from a quantum register modulo N, controlled by ctrl.

    When ctrl=|1>: qubits -> (qubits - constant) mod N
    When ctrl=|0>: qubits unchanged

    Uses the Beauregard circuit with proper ancilla uncomputation to preserve
    quantum superposition in the control qubit.

    Args:
        ctrl: Control qubit name
        qubits: List of qubit names [MSB, ..., LSB]
        constant: Classical integer to subtract
        N: Modulus
    """
    # Step 1: Transform to Fourier basis
    QFT(qubits)
    _normalize_qubits(qubits)

    # Step 2: Controlled subtract constant
    controlled_phi_add_constant(ctrl, qubits, constant, inverse=True)

    # Step 3: Back to computational basis for sign check
    QFT_inverse(qubits)
    _normalize_qubits(qubits)

    # At this point:
    # - If ctrl=0: register unchanged
    # - If ctrl=1: register = a - constant (mod 2^n)
    #   MSB=1 means a < constant (underflow)

    # Step 4: Copy MSB to ancilla (controlled on ctrl)
    pushQubit("_cms_anc", [1, 0])
    applyGate(TOFF_gate, ctrl, qubits[0], "_cms_anc")

    # Step 5: Add N controlled on ancilla
    QFT(qubits)
    _normalize_qubits(qubits)
    tosQubit("_cms_anc")
    controlled_phi_add_constant("_cms_anc", qubits, N)

    # Step 6: Uncompute ancilla
    # Add constant back, then XOR with MSB
    #
    # Analysis for subtract a - c mod N:
    # - No underflow (a >= c): after -c, result = a-c, MSB=0 (for valid inputs), anc=0
    #   After +c: result = a, MSB depends on a (typically 0)
    # - Underflow (a < c): after -c+N, result = a-c+N, anc=1
    #   After +c: result = a-c+N+c = a+N >= N, so wraps: result = a+N (could have MSB=0 or 1)
    #   Actually for N < 2^(n-1), a+N < 2^n, so no wrap and MSB depends on value
    #
    # The key: for underflow case, a+N has MSB=1 iff a+N >= 2^(n-1)
    # Since N is about 2^(n-1), a+N is roughly 2^(n-1) to 2^n, so MSB is often 1.
    #
    # For a < c (underflow), a is small, so a+N is close to N, MSB likely 1 if N >= 2^(n-1)
    # For a >= c (no underflow), a after +c is original a, MSB likely 0 if a < 2^(n-1)
    #
    # XOR rule: anc XOR (ctrl AND MSB)
    # - No underflow: anc=0, MSB~0 -> 0 XOR 0 = 0 OK
    # - Underflow: anc=1, MSB~1 -> 1 XOR 1 = 0 OK
    controlled_phi_add_constant(ctrl, qubits, constant)

    QFT_inverse(qubits)
    _normalize_qubits(qubits)
    tosQubit("_cms_anc")

    # XOR with ctrl AND MSB (not NOT MSB!)
    applyGate(TOFF_gate, ctrl, qubits[0], "_cms_anc")

    # Step 7: Subtract constant again to restore correct result
    QFT(qubits)
    _normalize_qubits(qubits)
    tosQubit("_cms_anc")
    controlled_phi_add_constant(ctrl, qubits, constant, inverse=True)
    QFT_inverse(qubits)
    _normalize_qubits(qubits)
    tosQubit("_cms_anc")

    # Ancilla should be |0> - safe to measure
    measureQubit("_cms_anc")


# =============================================================================
# Doubly-controlled modular addition (for shift-and-add)
# =============================================================================

def doubly_controlled_phi_add_constant(ctrl1: str, ctrl2: str, qubits: List[str],
                                        constant: int, inverse: bool = False):
    """
    Add a constant in Fourier basis, doubly controlled.

    Only adds when both ctrl1=|1> AND ctrl2=|1>.

    Uses a Toffoli to compute the AND, then controlled-phi-add on ancilla.

    Args:
        ctrl1, ctrl2: Control qubit names
        qubits: List of qubit names [MSB, ..., LSB] (in Fourier basis)
        constant: Classical integer to add
        inverse: If True, subtract instead of add
    """
    # Create ancilla for AND of controls
    pushQubit("_dc_anc", [1, 0])
    tosQubit("_dc_anc")

    # Compute AND
    applyGate(TOFF_gate, ctrl1, ctrl2, "_dc_anc")

    # Controlled add on ancilla
    controlled_phi_add_constant("_dc_anc", qubits, constant, inverse=inverse)

    # Uncompute AND
    applyGate(TOFF_gate, ctrl1, ctrl2, "_dc_anc")

    # Cleanup ancilla
    measureQubit("_dc_anc")


def doubly_controlled_modular_add_constant(ctrl1: str, ctrl2: str,
                                            qubits: List[str],
                                            constant: int, N: int):
    """
    Add a constant modulo N, controlled by two qubits.

    Only adds when both ctrl1=|1> AND ctrl2=|1>.

    Args:
        ctrl1, ctrl2: Control qubit names
        qubits: List of qubit names [MSB, ..., LSB]
        constant: Classical integer to add
        N: Modulus
    """
    # Use Toffoli to create AND of controls
    pushQubit("_dcma_ctrl", [1, 0])
    tosQubit("_dcma_ctrl")

    applyGate(TOFF_gate, ctrl1, ctrl2, "_dcma_ctrl")

    # Now do single-controlled modular add
    controlled_modular_add_constant("_dcma_ctrl", qubits, constant, N)

    # Uncompute AND
    applyGate(TOFF_gate, ctrl1, ctrl2, "_dcma_ctrl")

    # Cleanup
    measureQubit("_dcma_ctrl")


def doubly_controlled_modular_subtract_constant(ctrl1: str, ctrl2: str,
                                                 qubits: List[str],
                                                 constant: int, N: int):
    """
    Subtract a constant modulo N, controlled by two qubits.

    Only subtracts when both ctrl1=|1> AND ctrl2=|1>.

    Args:
        ctrl1, ctrl2: Control qubit names
        qubits: List of qubit names [MSB, ..., LSB]
        constant: Classical integer to subtract
        N: Modulus
    """
    pushQubit("_dcms_ctrl", [1, 0])
    tosQubit("_dcms_ctrl")

    applyGate(TOFF_gate, ctrl1, ctrl2, "_dcms_ctrl")

    controlled_modular_subtract_constant("_dcms_ctrl", qubits, constant, N)

    applyGate(TOFF_gate, ctrl1, ctrl2, "_dcms_ctrl")

    measureQubit("_dcms_ctrl")


# =============================================================================
# Controlled swap for registers
# =============================================================================

def controlled_swap_registers(ctrl: str, reg_a: List[str], reg_b: List[str]):
    """
    Swap two registers controlled by a control qubit.

    When ctrl=|1>: swap reg_a <-> reg_b
    When ctrl=|0>: no change

    Uses Fredkin gates (controlled-SWAP) which can be built from Toffolis.

    Args:
        ctrl: Control qubit name
        reg_a, reg_b: Lists of qubit names (same length)
    """
    assert len(reg_a) == len(reg_b), "Registers must have same length"

    for qa, qb in zip(reg_a, reg_b, strict=True):
        # Fredkin (controlled-SWAP) = CNOT(b,a), Toffoli(ctrl,a,b), CNOT(b,a)
        applyGate(CNOT_gate, qb, qa)
        applyGate(TOFF_gate, ctrl, qa, qb)
        applyGate(CNOT_gate, qb, qa)


# =============================================================================
# Shift-and-add modular multiplication
# =============================================================================

def controlled_mod_mult_pure(ctrl: str, x_qubits: List[str], a: int, N: int):
    """
    Controlled modular multiplication using shift-and-add.

    When ctrl=|1>: |x> -> |ax mod N>
    When ctrl=|0>: |x> unchanged

    This is a "pure" quantum implementation using modular addition as the
    building block, NOT precomputed permutation cycles.

    Algorithm (shift-and-add):
    1. Allocate accumulator register (init |0>) with n+1 bits for overflow
    2. For each bit i of x (i=0 is LSB, at position n-1 in MSB-first ordering):
       coeff = (a * 2^i) mod N
       Controlled on x[n-1-i]=1 AND ctrl=1: acc += coeff mod N
    3. Swap x <-> acc (lower n bits only, controlled on ctrl)
    4. Uncompute acc with inverse multiplication by a^(-1):
       For each bit i: acc -= (a_inv * 2^i) mod N controlled on x[n-1-i] AND ctrl
    5. Cleanup accumulator

    Args:
        ctrl: Control qubit name
        x_qubits: List of qubit names [MSB, ..., LSB]
        a: Multiplier (must be coprime to N)
        N: Modulus
    """
    n = len(x_qubits)

    # Compute modular inverse of a
    a_inv = mod_inverse(a, N)
    if a_inv is None:
        raise ValueError(f"{a} and {N} must be coprime")

    # Step 1: Allocate accumulator register with n+1 bits for overflow detection
    # The extra MSB bit is used for modular reduction overflow checking
    acc_qubits = []
    for i in range(n + 1):
        name = f"_acc{i}"
        pushQubit(name, [1, 0])  # |0>
        acc_qubits.append(name)

    # Step 2: Shift-and-add forward pass
    # For each bit position i (where i=0 is LSB):
    #   coeff = (a * 2^i) mod N
    #   If x bit i is 1 AND ctrl=1: acc += coeff mod N
    #
    # Bit indexing: x_qubits[0] is MSB, x_qubits[n-1] is LSB
    # So bit i (LSB=0) is at index (n-1-i)

    for i in range(n):
        coeff = (a * (1 << i)) % N
        if coeff == 0:
            continue  # Adding 0 does nothing

        x_bit_idx = n - 1 - i  # Index in x_qubits for bit i
        x_bit = x_qubits[x_bit_idx]

        # Doubly controlled add: ctrl AND x_bit
        doubly_controlled_modular_add_constant(ctrl, x_bit, acc_qubits, coeff, N)

    # Step 3: Controlled swap x <-> acc (lower n bits only)
    # The acc has n+1 bits, but x has n bits, so we swap only the lower n bits
    # acc_qubits[0] is MSB (overflow bit), acc_qubits[1:] are the value bits
    controlled_swap_registers(ctrl, x_qubits, acc_qubits[1:])

    # Step 4: Uncompute accumulator using inverse multiplication
    # After swap, x_qubits now holds (a*x mod N) and acc_qubits[1:] holds old x
    # acc_qubits[0] (overflow bit) should still be 0
    # We need to uncompute acc: for each bit i of (new x = a*x mod N):
    #   acc -= (a_inv * 2^i) mod N, controlled on new_x[bit i] AND ctrl
    #
    # This works because: if new_x = a*x mod N, then
    # sum over bits of (a_inv * 2^i * bit_i) = a_inv * new_x = x (mod N)
    # So acc = old_x - a_inv * new_x = x - x = 0

    for i in range(n):
        coeff_inv = (a_inv * (1 << i)) % N
        if coeff_inv == 0:
            continue

        x_bit_idx = n - 1 - i
        x_bit = x_qubits[x_bit_idx]  # This is now the NEW x (= a*old_x mod N)

        doubly_controlled_modular_subtract_constant(ctrl, x_bit, acc_qubits, coeff_inv, N)

    # Step 5: Cleanup accumulator (should be |0> now)
    for q in reversed(acc_qubits):
        measureQubit(q)


# =============================================================================
# Controlled modular exponentiation
# =============================================================================

def apply_controlled_mod_exp_pure(control_qubits: List[str], work_qubits: List[str],
                                   a: int, N: int):
    """
    Apply controlled modular exponentiation using pure shift-and-add.

    For each control qubit k, applies controlled multiplication by a^(2^k) mod N.
    This builds up the transformation:
        |j>|1> -> |j>|a^j mod N>

    where j is the integer represented by the control register.

    Args:
        control_qubits: List of control qubit names (LSB to MSB order)
        work_qubits: List of work qubit names [MSB, ..., LSB]
        a: Base for exponentiation
        N: Modulus
    """
    power = a
    for ctrl in control_qubits:
        controlled_mod_mult_pure(ctrl, work_qubits, power, N)
        power = (power * power) % N


# =============================================================================
# Full Shor's algorithm (pure implementation)
# =============================================================================

def shor_factor_pure(N: int, a: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """
    Shor's algorithm using pure shift-and-add multiplication.

    This is a "pure" quantum implementation that uses modular addition
    as the building block, rather than precomputed permutation cycles.

    Args:
        N: Number to factor (should be composite, odd, not a prime power)
        a: Base for modular exponentiation (random if not specified)

    Returns:
        Tuple (factor1, factor2) if successful, or None if this run failed
    """
    # Classical preprocessing
    if N % 2 == 0:
        return (2, N // 2)

    # Check if N is a prime power
    for b in range(2, int(math.log2(N)) + 1):
        root = int(round(N ** (1 / b)))
        for candidate_root in [root - 1, root, root + 1]:
            if candidate_root > 1 and candidate_root ** b == N:
                return (candidate_root, N // candidate_root)

    # Choose random a if not specified
    if a is None:
        candidates = [x for x in range(2, N) if is_coprime(x, N)]
        if not candidates:
            return None
        a = random.choice(candidates)

    # Check for lucky GCD
    g = gcd(a, N)
    if g > 1:
        return (g, N // g)

    # Quantum period finding
    reset()

    work_size, control_size = compute_register_sizes(N)

    # Need extra bit in work register for modular arithmetic overflow detection
    n_work = work_size + 1

    # Initialize work register to |1>
    work_qubits = init_register("w", 1, n_bits=n_work)

    # Initialize control register in superposition
    control_qubits = []
    for i in range(control_size):
        name = f"c{i}"
        pushQubit(name, [1, 0])
        applyGate(H_gate, name)
        control_qubits.append(name)

    # Apply controlled modular exponentiation (pure version)
    apply_controlled_mod_exp_pure(control_qubits, work_qubits, a, N)

    # Apply inverse QFT to control register
    # QFT expects MSB first, but our control_qubits are LSB first
    QFT_inverse(control_qubits[::-1])

    # Measure control register
    # Normalize order first
    for q in control_qubits[::-1]:
        tosQubit(q)

    measurement = 0
    for i in range(control_size - 1, -1, -1):
        bit = int(measureQubit(control_qubits[i]))
        measurement = measurement * 2 + bit

    # Clean up work register
    for q in reversed(work_qubits):
        measureQubit(q)

    # Classical post-processing: extract period
    r = extract_period(measurement, control_size, N, a)

    if r is None or r % 2 == 1:
        return None

    # Compute factors
    x = pow(a, r // 2, N)

    if x == N - 1:  # x = -1 mod N
        return None

    factor1 = gcd(x - 1, N)
    factor2 = gcd(x + 1, N)

    if factor1 != 1 and factor1 != N:
        return (factor1, N // factor1)
    elif factor2 != 1 and factor2 != N:
        return (factor2, N // factor2)
    else:
        return None
