import functools
import jax
import jax.numpy as jnp
from jax import lax

OMEGA_SMALL_EPS: float = 1e-9


def _precompute_structure_factors_all(
    widths: jax.Array,
    kappas: jax.Array,
    omegas: jax.Array,
    block_size: int,
) -> jax.Array:
    """
    Computes structure factors for all channels simultaneously.

    Args:
        widths: (N,) array of domain widths.
        kappas: (C, N) array of kappa values for each channel.
        omegas: (C,) array of frequency mismatches for each channel.
        block_size: integer used for grouping.

    Returns:
        factors: (C, N // block_size) array of structure factors.
    """
    # 1. Compute cumulative positions
    # z_nodes[0] = 0, z_nodes[i] = sum(w[:i])
    # Shape: (N+1,)
    z_nodes = jnp.pad(jnp.cumsum(widths), (1, 0), constant_values=0.0)

    # 2. Compute exponentials
    # Shape: (C, N+1)
    # Broadcast omegas against z_nodes
    phases = 1j * omegas[:, None] * z_nodes[None, :]
    exps = jnp.exp(phases)

    # 3. Compute domain integrals
    # integral = kappa * (exp(i w z_{n+1}) - exp(i w z_n)) / (i w)
    # Shape: (C, N)
    diffs = exps[:, 1:] - exps[:, :-1]

    # Handle omega close to 0
    # Shape: (C, 1) to broadcast against (C, N)
    inv_iw = 1.0 / (1j * omegas[:, None])

    # Standard case
    vals_main = kappas * diffs * inv_iw

    # Small omega case: integral -> kappa * width
    # widths broadcasts to (C, N)
    vals_small = kappas * widths[None, :]

    # Select based on omega magnitude
    # is_small: (C, 1)
    is_small = jnp.abs(omegas[:, None]) < OMEGA_SMALL_EPS
    factors_domains = jnp.where(is_small, vals_small, vals_main)

    # 4. Sum into blocks
    # Reshape to (C, n_blocks, block_size) and sum over last axis
    n_blocks = widths.shape[0] // block_size
    # C is likely 4
    C = kappas.shape[0]
    factors_reshaped = factors_domains.reshape(C, n_blocks, block_size)
    factors_blocks = jnp.sum(factors_reshaped, axis=2)

    return factors_blocks


def _lfaga_step_update(
    a_curr: jax.Array,
    f_factors: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> jax.Array:
    """
    Updates envelopes A using the LFAGA scheme with precomputed factors.
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
def _lfaga_scan_loop(
    a_initial: jax.Array,
    f_shg_series: jax.Array,
    f_sfg_series: jax.Array,
    f_neg_shg_series: jax.Array,
    f_neg_sfg_series: jax.Array,
    *,
    return_trace: bool,
) -> tuple[jax.Array, jax.Array | None]:
    def scan_body(a_curr: jax.Array, factors: tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array | None]:
        a_next = _lfaga_step_update(a_curr, factors)
        track_val = a_next if return_trace else None
        return a_next, track_val

    # Zip the factors for scanning
    xs = (f_shg_series, f_sfg_series, f_neg_shg_series, f_neg_sfg_series)

    a_final, a_stacked = lax.scan(scan_body, a_initial, xs, unroll=20)

    if return_trace and a_stacked is not None:
        full_trace = jnp.vstack([a_initial[None, :], a_stacked])
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
    Optimized for binary waves, using vectorized parallel structure factor computation.
    """
    _validate_inputs(domain_widths, kappa_shg_vals, kappa_sfg_vals, a_initial)

    # Pad inputs if necessary
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

    # Prepare stacked inputs for single-pass precomputation
    # Channels: 0: SHG(+), 1: SFG(+), 2: SHG(-), 3: SFG(-)
    kappas_stack = jnp.stack([k_shg_new, k_sfg_new, k_shg_new, k_sfg_new])
    omegas_stack = jnp.array([delta_k1, delta_k2, -delta_k1, -delta_k2])

    # Compute all factors in one go
    # Shape: (4, N_blocks)
    factors_all = _precompute_structure_factors_all(widths_new, kappas_stack, omegas_stack, block_size)

    # Unpack
    f_shg = factors_all[0]
    f_sfg = factors_all[1]
    f_neg_shg = factors_all[2]
    f_neg_sfg = factors_all[3]

    # Run scan
    a_final, _ = _lfaga_scan_loop(a_initial, f_shg, f_sfg, f_neg_shg, f_neg_sfg, return_trace=False)

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
    """
    _validate_inputs(domain_widths, kappa_shg_vals, kappa_sfg_vals, a_initial)

    # Pad inputs if necessary
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

    # Prepare stacked inputs
    kappas_stack = jnp.stack([k_shg_new, k_sfg_new, k_shg_new, k_sfg_new])
    omegas_stack = jnp.array([delta_k1, delta_k2, -delta_k1, -delta_k2])

    # Compute all factors
    factors_all = _precompute_structure_factors_all(widths_new, kappas_stack, omegas_stack, block_size)

    f_shg = factors_all[0]
    f_sfg = factors_all[1]
    f_neg_shg = factors_all[2]
    f_neg_sfg = factors_all[3]

    # Run scan
    a_final, trace = _lfaga_scan_loop(a_initial, f_shg, f_sfg, f_neg_shg, f_neg_sfg, return_trace=True)

    return a_final, trace
