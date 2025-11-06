import jax
import jax.numpy as jnp


# --- Helper Functions for Analytical Calculation ---
def _r(a: jax.Array, b: jax.Array, k: jax.Array) -> jax.Array:
    """
    Calculates R([a,b]; k) = ∫[a,b] exp(-ikz) dz
    Uses the numerically stable sinc formulation from the theory,
    which is inherently grad-safe and handles k=0.

    R = exp(-ik(a+b)/2) * l * sinc(kl/2)
    """
    ell = b - a
    center = (a + b) / 2.0

    # jnp.sinc(x) = sin(pi*x)/(pi*x)
    # We need unnormalized sinc(y) where y = k*ell/2.
    # We set x = y/pi = k*ell/(2*pi)
    sinc_arg = k * ell / (2.0 * jnp.pi)

    # jnp.sinc handles the k=0 case (where sinc_arg=0) automatically
    sinc_val = jnp.sinc(sinc_arg)

    return jnp.exp(-1j * k * center) * ell * sinc_val


def _j(a: jax.Array, b: jax.Array, dk1: jax.Array, dk2: jax.Array) -> jax.Array:
    """
    Calculates J([a,b]; dk1, dk2)
    Handles singularities at dk1=0 and/or dk2=0 in a grad-safe way.
    """
    ell = b - a

    # --- Create safe denominators for all *external* divisions ---
    # These prevent NaN propagation during gradient calculations.
    dk1_safe = jnp.where(dk1 == 0.0, 1.0, dk1)
    dk2_safe = jnp.where(dk2 == 0.0, 1.0, dk2)

    # Note: _r calls are now inherently safe due to the sinc formulation.
    r_dk2 = _r(a, b, dk2)
    r_dk1_dk2 = _r(a, b, dk1 + dk2)

    # --- General case: dk1 != 0 ---
    # Division by dk1_safe is now safe.
    j_general = (jnp.exp(-1j * dk1 * a) * r_dk2 - r_dk1_dk2) / (1j * dk1_safe)

    # --- Handle cases where dk1 = 0 ---

    # Subcase: dk1 = 0, dk2 = 0
    # J([a,b]; 0, 0) = ℓ²/2
    j_dk1_zero_dk2_zero = ell**2 / 2.0

    # Subcase: dk1 = 0, dk2 != 0
    # J([a,b]; 0, dk2) = ∫[a,b] (z-a)exp(-i*dk2*z) dz
    # Solved via integration by parts, using safe denominators.
    term1 = 1j * ell * jnp.exp(-1j * dk2 * b) / dk2_safe
    term2 = (jnp.exp(-1j * dk2 * b) - jnp.exp(-1j * dk2 * a)) / (dk2_safe**2)
    j_dk1_zero_dk2_nonzero = term1 + term2

    # Combine dk1=0 subcases
    j_dk1_zero = jnp.where(dk2 == 0, j_dk1_zero_dk2_zero, j_dk1_zero_dk2_nonzero)

    # Final selection based on dk1
    return jnp.where(dk1 == 0, j_dk1_zero, j_general)


# --- JIT-Compiled Analytical Calculation ---
@jax.jit
def calc_s_analytical(kappas: jax.Array, widths: jax.Array, dk1: jax.Array, dk2: jax.Array) -> jax.Array:
    """
    Calculates the S-functional analytically using the O(N) discretized formula.
    This version is JIT-compiled and gradient-safe.
    """
    # Calculate domain boundaries [z_{n-1}, z_n] for each segment
    z_end = jnp.cumsum(widths)
    z_start = jnp.pad(z_end[:-1], (1, 0))

    # --- Calculate terms from Equation (*) ---

    # 1. Diagonal sum part: Σ_{n=1}^{N} κ_n² J_n(dk1, dk2)
    j_n = _j(z_start, z_end, dk1, dk2)
    diagonal_sum = jnp.sum(kappas**2 * j_n)

    # 2. Off-diagonal double sum part: Σ_{n=1}^{N} Σ_{m=1}^{n-1} ...
    # Implemented in O(N) using the cumulative sum trick.

    # s^(1)_n = κ_n R_n(Δk_1)
    # s^(2)_n = κ_n R_n(Δk_2)
    s1_n = kappas * _r(z_start, z_end, dk1)
    s2_n = kappas * _r(z_start, z_end, dk2)

    # Inner sum: C_n = Σ_{m=1}^{n} s^(1)_m
    inner_sum_cumulative = jnp.cumsum(s1_n)

    # Shift to get C_{n-1} = [0, s1_1, s1_1+s1_2, ...]
    inner_sum_shifted = jnp.pad(inner_sum_cumulative, (1, 0))[:-1]

    # Outer sum: Σ_{n=1}^{N} s^(2)_n * C_{n-1}
    double_sum = jnp.sum(s2_n * inner_sum_shifted)

    return double_sum + diagonal_sum
