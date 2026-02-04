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
    (Duplicated from _cthg_lfaga_binary.py to keep module self-contained)

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
    # C is likely 4 or 2 depending on usage
    C = kappas.shape[0]
    factors_reshaped = factors_domains.reshape(C, n_blocks, block_size)
    factors_blocks = jnp.sum(factors_reshaped, axis=2)

    return factors_blocks


def _magnus_step_update(
    u_curr: jax.Array,
    f_factors: tuple[jax.Array, jax.Array],
) -> jax.Array:
    # Updates normalized envelopes u using the Magnus-Cayley scheme.
    # Note: Using u = A (no normalization) based on 1:1 ODE structure.
    u1, u2, _ = u_curr
    f_shg, f_sfg = f_factors

    # A_n is same as u_n (normalized u = A)
    a1 = u1
    a2 = u2
    # a3 = u3 # Not needed for mu calculation

    # Compute mu elements (frozen at step n)
    # SHG: A1' ~ A2 A1*, A2' ~ A1^2
    # mu_12 = beta_shg * A1*
    mu_12 = f_shg * jnp.conj(a1)

    # SFG: A1' ~ A3 A2*, A2' ~ 2 A3 A1*, A3' ~ 3 A1 A2
    # mu_13 corresponding to A1-A3 link (via A2)
    # mu_13 = beta_sfg * A2*
    mu_13 = f_sfg * jnp.conj(a2)

    # mu_23 corresponding to A2-A3 link (via A1) with coeff 2
    # mu_23 = 2 * beta_sfg * A1*
    mu_23 = 2.0 * f_sfg * jnp.conj(a1)

    # Construct Omega matrix (Skew-Hermitian)
    # Omega = i * [[0, mu12, mu13], [mu12*, 0, mu23], [mu13*, mu23*, 0]]

    zero = jnp.complex64(0.0)

    # Row 0
    o_00 = zero
    o_01 = 1j * mu_12
    o_02 = 1j * mu_13

    # Row 1
    o_10 = 1j * jnp.conj(mu_12)
    o_11 = zero
    o_12 = 1j * mu_23

    # Row 2
    o_20 = 1j * jnp.conj(mu_13)
    o_21 = 1j * jnp.conj(mu_23)
    o_22 = zero

    omega = jnp.array([[o_00, o_01, o_02], [o_10, o_11, o_12], [o_20, o_21, o_22]])

    # Cayley Transform Update
    # u_next = (I - 0.5*Omega)^(-1) (I + 0.5*Omega) u_curr

    eye = jnp.eye(3, dtype=jnp.complex64)
    half_omega = 0.5 * omega

    mat_plus = eye + half_omega
    mat_minus = eye - half_omega

    rhs = mat_plus @ u_curr

    # Solve (I - 0.5*Omega) u_next = rhs
    u_next = jnp.linalg.solve(mat_minus, rhs)

    return u_next


@functools.partial(jax.jit, static_argnames=["return_trace"])
def _magnus_scan_loop(
    u_initial: jax.Array,
    f_shg_series: jax.Array,
    f_sfg_series: jax.Array,
    *,
    return_trace: bool,
) -> tuple[jax.Array, jax.Array | None]:
    def scan_body(u_curr: jax.Array, factors: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array | None]:
        u_next = _magnus_step_update(u_curr, factors)

        # If tracing, we might want to convert back to A for storage?
        # Or store u. Let's store u for now, convert outside if needed.
        # But for 'simulate_magnus' return value, we definitely need conversion.
        # Let's just track u.
        track_val = u_next if return_trace else None
        return u_next, track_val

    # Zip the factors for scanning
    xs = (f_shg_series, f_sfg_series)

    u_final, u_stacked = lax.scan(scan_body, u_initial, xs, unroll=20)

    if return_trace and u_stacked is not None:
        full_trace = jnp.vstack([u_initial[None, :], u_stacked])
        return u_final, full_trace

    return u_final, None


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


def simulate_magnus(  # noqa: PLR0913
    domain_widths: jax.Array,
    kappa_shg_vals: jax.Array,
    kappa_sfg_vals: jax.Array,
    delta_k1: jax.Array,
    delta_k2: jax.Array,
    a_initial: jax.Array,
    block_size: int = 1,
) -> jax.Array:
    """
    Simulates CTHG using the Quasi-Magnus Cayley scheme.
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
    # We only need positive Delta k factors for Magnus mu definitions
    # Channels: 0: SHG(+), 1: SFG(+)
    kappas_stack = jnp.stack([k_shg_new, k_sfg_new])
    omegas_stack = jnp.array([delta_k1, delta_k2])

    # Compute factors
    # Shape: (2, N_blocks)
    factors_all = _precompute_structure_factors_all(widths_new, kappas_stack, omegas_stack, block_size)

    f_shg = factors_all[0]
    f_sfg = factors_all[1]

    # Convert A_initial to u_initial
    # Using u = A
    u_initial = a_initial

    # Run scan
    u_final, _ = _magnus_scan_loop(u_initial, f_shg, f_sfg, return_trace=False)

    # Convert u_final back to A_final
    # u = A
    a_final = u_final

    return a_final


def simulate_magnus_with_trace(  # noqa: PLR0913
    domain_widths: jax.Array,
    kappa_shg_vals: jax.Array,
    kappa_sfg_vals: jax.Array,
    delta_k1: jax.Array,
    delta_k2: jax.Array,
    a_initial: jax.Array,
    block_size: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """
    Simulates CTHG using the Quasi-Magnus Cayley scheme and returns trace.
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
    kappas_stack = jnp.stack([k_shg_new, k_sfg_new])
    omegas_stack = jnp.array([delta_k1, delta_k2])

    # Compute factors
    factors_all = _precompute_structure_factors_all(widths_new, kappas_stack, omegas_stack, block_size)

    f_shg = factors_all[0]
    f_sfg = factors_all[1]

    # Convert A_initial to u_initial
    # Using u = A
    u_initial = a_initial

    # Run scan
    u_final, trace = _magnus_scan_loop(u_initial, f_shg, f_sfg, return_trace=True)

    # Convert u_final back to A_final
    # u = A
    a_final = u_final

    return a_final, trace
