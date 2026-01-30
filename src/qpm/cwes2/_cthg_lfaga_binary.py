import functools

import jax
import jax.numpy as jnp
from jax import lax

OMEGA_SMALL_EPS: float = 1e-9


def _calc_global_structure_factors_pair(
    widths_block: jax.Array,
    kappa_block: jax.Array,
    omega: jax.Array,
    z_global: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Computes global Phase-Weighted Integrals F[kappa](omega) and F[kappa](-omega)
    simultaneously using conjugate symmetry.

    F_global(w) = exp(i w z_global) * F_local(w)
                = exp(i w z(0)) * integral_{0}^{h} kappa(z') exp(i w z') dz'
                = integral_{0}^{h} kappa(z') exp(i w (z_global + z')) dz'

    Returns:
        f_pos: F[kappa](omega)
        f_neg: F[kappa](-omega)
    """
    # 1. Compute global boundaries: [z_g, z_g+z1, ..., z_g+h]
    # widths_block shape: (M,)
    # boundaries shape: (M+1,)
    # Note: jnp.cumsum is inclusive.
    local_boundaries = jnp.pad(jnp.cumsum(widths_block), (1, 0), constant_values=0.0)
    global_boundaries = local_boundaries + z_global

    # 2. Compute exponentials: exp(i * omega * global_boundaries)
    # This is the most expensive step.
    # shape: (M+1,)
    phases = 1j * omega * global_boundaries
    exps_pos = jnp.exp(phases)

    # 3. Differences: exp(i w Z_{j+1}) - exp(i w Z_j)
    # shape: (M,)
    diffs_pos = exps_pos[1:] - exps_pos[:-1]

    # 4. Weighted sums
    # Positive freq
    sum_pos = jnp.dot(kappa_block, diffs_pos)

    # Negative freq (-omega)
    # exp(-i w z) = conj(exp(i w z)) => diffs(-w) = conj(diffs(w))
    # We use conj(diffs_pos)
    diffs_neg = jnp.conj(diffs_pos)
    sum_neg = jnp.dot(kappa_block, diffs_neg)

    # 5. Normalize by +/- i*omega
    # Handle small omega case
    # If partial(jax.jit) is used, this branching is symbolic.

    # Precompute factors
    inv_iw = 1.0 / (1j * omega)

    val_pos = sum_pos * inv_iw
    val_neg = sum_neg * (-inv_iw)  # divide by -iw is * -1/iw

    # Small omega fallback (integral of kappa)
    # limit sum_pos / iw -> sum(kappa * h)
    val_zero = jnp.dot(kappa_block, widths_block)

    # Use 'where' for safety
    is_small = jnp.abs(omega) < OMEGA_SMALL_EPS
    f_pos = jnp.where(is_small, val_zero, val_pos)
    f_neg = jnp.where(is_small, val_zero, val_neg)

    return f_pos, f_neg


def _lfaga_step_update(
    a_curr: jax.Array,
    f_factors: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> jax.Array:
    """
    Updates envelopes A using the LFAGA scheme.
    """
    a1, a2, a3 = a_curr
    f_shg, f_sfg, f_neg_shg, f_neg_sfg = f_factors

    # A1' = i [ A2 A1* F(dk_shg) + A3 A2* F(dk_sfg) ]
    term1 = 1j * (a2 * jnp.conj(a1) * f_shg + a3 * jnp.conj(a2) * f_sfg)

    # A2' = i [ A1^2 F(-dk_shg) + 2 A3 A1* F(dk_sfg) ]
    term2 = 1j * ((a1**2) * f_neg_shg + 2 * a3 * jnp.conj(a1) * f_sfg)

    # A3' = i [ 3 A_1 A_2 F(-dk_sfg) ]
    term3 = 1j * (3 * a1 * a2 * f_neg_sfg)

    return jnp.array([a1 + term1, a2 + term2, a3 + term3])


@functools.partial(jax.jit, static_argnames=["return_trace"])
def _lfaga_kernel(  # noqa: PLR0913
    widths_padded: jax.Array,
    kappa_shg_padded: jax.Array,
    kappa_sfg_padded: jax.Array,
    delta_k1: jax.Array,
    delta_k2: jax.Array,
    a_initial: jax.Array,
    *,
    return_trace: bool,
) -> tuple[jax.Array, jax.Array | None]:
    # scan over blocks
    def scan_body(
        carry: tuple[jax.Array, jax.Array], block_data: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array | None]:
        a_curr, z_global = carry
        w_block, k_shg_block, k_sfg_block = block_data

        h_total = jnp.sum(w_block)

        # 1. Compute Global Structure Factors directly
        # Returns (F[w], F[-w]) pairs
        f_shg, f_neg_shg = _calc_global_structure_factors_pair(w_block, k_shg_block, delta_k1, z_global)
        f_sfg, f_neg_sfg = _calc_global_structure_factors_pair(w_block, k_sfg_block, delta_k2, z_global)

        # 2. Update Step
        f_factors = (f_shg, f_sfg, f_neg_shg, f_neg_sfg)
        a_next = _lfaga_step_update(a_curr, f_factors)

        # Update global z
        z_next = z_global + h_total

        new_carry = (a_next, z_next)
        track_val = a_next if return_trace else None
        return new_carry, track_val

    scan_data = (widths_padded, kappa_shg_padded, kappa_sfg_padded)
    init_carry = (a_initial, jnp.array(0.0))

    (a_final, _), a_stacked = lax.scan(scan_body, init_carry, scan_data, unroll=20)

    if return_trace and a_stacked is not None:
        full_trace = jnp.vstack([a_initial, a_stacked])
        return a_final, full_trace

    return a_final, None


def _validate_inputs(
    domain_widths: jax.Array,
    kappa_shg: jax.Array,
    kappa_sfg: jax.Array,
    a_initial: jax.Array,
) -> None:
    if (
        domain_widths.ndim != 1
        or kappa_shg.ndim != 1
        or kappa_sfg.ndim != 1
        or domain_widths.shape != kappa_shg.shape
        or domain_widths.shape != kappa_sfg.shape
    ):
        msg = "domain_widths and kappa_vals must be 1D arrays of the same shape."
        raise ValueError(msg)

    if a_initial.shape != (3,) or (a_initial.dtype not in (jnp.complex64, jnp.complex128)):
        msg = "a_initial must be a 1D array of shape (3,) and complex dtype."
        raise ValueError(msg)


def simulate_lfaga(  # noqa: PLR0913
    domain_widths: jax.Array,
    kappa_shg_vals: jax.Array,
    kappa_sfg_vals: jax.Array,
    delta_k1: jax.Array,
    delta_k2: jax.Array,
    a_initial: jax.Array,
    block_size: int = 1,
) -> jax.Array:
    """
    Simulates CTHG using the LFAGA (Longitudinal Fourier Averaged Global Approximation) method.
    Optimized for binary (piecewise constant) waves.
    """
    _validate_inputs(domain_widths, kappa_shg_vals, kappa_sfg_vals, a_initial)

    # Pad to multiple of block_size
    n_domains = domain_widths.shape[0]
    remainder = n_domains % block_size
    if remainder != 0:
        pad_len = block_size - remainder
        widths_new = jnp.pad(domain_widths, (0, pad_len), constant_values=0.0)
        k_shg_new = jnp.pad(kappa_shg_vals, (0, pad_len), constant_values=0.0)
        k_sfg_new = jnp.pad(kappa_sfg_vals, (0, pad_len), constant_values=0.0)
    else:
        widths_new = domain_widths
        k_shg_new = kappa_shg_vals
        k_sfg_new = kappa_sfg_vals

    n_blocks = widths_new.shape[0] // block_size
    w_matrix = widths_new.reshape(n_blocks, block_size)
    k_shg_matrix = k_shg_new.reshape(n_blocks, block_size)
    k_sfg_matrix = k_sfg_new.reshape(n_blocks, block_size)

    a_final, _ = _lfaga_kernel(w_matrix, k_shg_matrix, k_sfg_matrix, delta_k1, delta_k2, a_initial, return_trace=False)

    return a_final


def simulate_lfaga_with_trace(  # noqa: PLR0913
    domain_widths: jax.Array,
    kappa_shg_vals: jax.Array,
    kappa_sfg_vals: jax.Array,
    delta_k1: jax.Array,
    delta_k2: jax.Array,
    a_initial: jax.Array,
    block_size: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """
    Simulates CTHG using the LFAGA method and returns trace.
    Optimized for binary (piecewise constant) waves.
    """
    _validate_inputs(domain_widths, kappa_shg_vals, kappa_sfg_vals, a_initial)

    n_domains = domain_widths.shape[0]
    remainder = n_domains % block_size
    if remainder != 0:
        pad_len = block_size - remainder
        widths_new = jnp.pad(domain_widths, (0, pad_len), constant_values=0.0)
        k_shg_new = jnp.pad(kappa_shg_vals, (0, pad_len), constant_values=0.0)
        k_sfg_new = jnp.pad(kappa_sfg_vals, (0, pad_len), constant_values=0.0)
    else:
        widths_new = domain_widths
        k_shg_new = kappa_shg_vals
        k_sfg_new = kappa_sfg_vals

    n_blocks = widths_new.shape[0] // block_size
    w_matrix = widths_new.reshape(n_blocks, block_size)
    k_shg_matrix = k_shg_new.reshape(n_blocks, block_size)
    k_sfg_matrix = k_sfg_new.reshape(n_blocks, block_size)

    a_final, trace = _lfaga_kernel(w_matrix, k_shg_matrix, k_sfg_matrix, delta_k1, delta_k2, a_initial, return_trace=True)

    return a_final, trace
