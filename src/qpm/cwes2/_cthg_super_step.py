import functools

import jax
import jax.numpy as jnp
from jax import lax

OMEGA_SMALL_EPS: float = 1e-9


def _get_lin(delta_k1: jax.Array, delta_k2: jax.Array) -> jax.Array:
    return jnp.array([0.0, delta_k1 * 1j, (delta_k1 + delta_k2) * 1j], dtype=jnp.complex64)


def _phi_optimized(exp_val: jax.Array, omega: jax.Array, h: jax.Array) -> jax.Array:
    """Computes (exp(i*omega*h) - 1) / (i*omega) safely."""
    # Note: The analytical formula phi = (e^{iwh} - 1)/(iw).
    # Here input `omega` is effectively `w` (no 'i' factor included in argument name, typically).
    # However, looking at _cthg_perturbation, the omega passes are:
    # omega_a = lin2 = i*dk1.
    # So the 'omega' arg here INCLUDES the 'i'.
    # Let's verify `val_large = (exp_val - 1.0) / omega`.
    # If omega = i*my_w, this is (exp(i*my_w*h) - 1) / (i*my_w). Correct.

    val_small = h + h**2 * omega / 2.0
    val_large = (exp_val - 1.0) / omega
    return jnp.where(jnp.abs(omega) < OMEGA_SMALL_EPS, val_small, val_large)


def _calc_structure_factors(
    widths_block: jax.Array,
    kappa_shg_block: jax.Array,
    kappa_sfg_block: jax.Array,
    delta_k_shg: jax.Array,
    delta_k_sfg: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Computes Structure Factors Psi(omega) for the block directly incorporating kappa.
    Psi(omega) = sum_j [ kappa_j * exp(omega * Z_j) * phi(omega, h_j) ]
    Note: omega here includes the imaginary unit 'i'.
    """
    # Cumulative depths Z_j. Z_0 = 0.
    # widths_block shape: (M,)
    # Z shape: (M,)
    z_cumulative = jnp.cumsum(jnp.pad(widths_block[:-1], (1, 0), constant_values=0.0))

    # Define omegas (including 'i').
    w_shg = 1j * delta_k_shg
    w_sfg = 1j * delta_k_sfg

    def get_psi(w: jax.Array, kappa_block: jax.Array) -> jax.Array:
        # exp_wh shape: (M,)
        exp_wh = jnp.exp(w * widths_block)
        # phi shape: (M,)
        phi = _phi_optimized(exp_wh, w, widths_block)
        # term shape: (M,)
        term = kappa_block * jnp.exp(w * z_cumulative) * phi
        return jnp.sum(term)

    psi_shg = get_psi(w_shg, kappa_shg_block)  # For Delta k_SHG (uses kappa_shg)
    psi_sfg = get_psi(w_sfg, kappa_sfg_block)  # For Delta k_SFG (uses kappa_sfg)

    # For negative k, we use the CONJUGATE of kappa?
    # Original derivation:
    # d B1 / dz = i k1 exp(-ildk z) B2 ...
    # Wait, the structure factor derivation usually pulls kappa out.
    # If we put kappa IN, we must handle the fact that B equations use kappa or kappa*.
    # Original code:
    # v1 = |k_shg| B1* B2 Psi(dk_shg) ...
    # where Psi uses s_j.
    # Now we want Psi to include kappa_j.
    # If kappa is real, kappa* = kappa.
    # If kappa is complex, we need to be careful.
    # v1 term ~ kappa * exp(...)
    # v2 term ~ kappa * exp(...) or kappa* * exp(...)?
    # Let's check `_super_step_update` in original code.
    # v1 = k_mag_shg * ... * psi_shg. psi_shg scales with s_j. So v1 scales with k_mag * s_j = kappa_j.
    # v2 = k_mag_shg * ... * psi_neg_shg. psi_neg_shg scales with s_j. So v2 scales with kappa_j.
    # Wait, usually for SHG/SFG:
    # dA1/dz ~ d_eff * A1* A2 exp(-idk z) -> involve d_eff.
    # dA2/dz ~ d_eff * A1^2 exp(idk z) -> involve d_eff.
    # In QPM, usually d_eff changes sign.
    # So both terms use d_eff (or kappa).
    # So we can just use kappa for both +dk and -dk terms.
    # If kappa is strictly real (just periodic sign), then Yes.
    # If kappa is complex, typically K and K* appear in Hamiltonian.
    # Checking standard TWM equations:
    # A1' = i K1 A3 A2* exp(-idk z)
    # A2' = i K2 A3 A1* exp(-idk z)
    # A3' = i K3 A1 A2 exp(idk z)
    # If we assume conservation and real chi2, then K proportional to chi2.
    # So it uses chi2 for all.
    # So we use kappa_block for all, not conjugate.

    psi_neg_shg = get_psi(-w_shg, kappa_shg_block)  # For -Delta k_SHG
    psi_neg_sfg = get_psi(-w_sfg, kappa_sfg_block)  # For -Delta k_SFG

    return psi_shg, psi_sfg, psi_neg_shg, psi_neg_sfg


def _super_step_update(
    b_curr: jax.Array,
    h_total: jax.Array,
    psi_factors: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    lin: jax.Array,
) -> jax.Array:
    """
    B_NL_block = i * exp(L*H) * Vector(...)
    Vector components involve kappa-weighted Psi factors.
    """
    b1, b2, b3 = b_curr
    psi_shg, psi_sfg, psi_neg_shg, psi_neg_sfg = psi_factors

    # Construct the perturbation vector vector_v
    # Row 1: B1* B2 Psi(dk_shg) + B2* B3 Psi(dk_sfg)
    # (Previously multiplied by |kappa|, now Psi includes kappa)
    v1 = jnp.conj(b1) * b2 * psi_shg + jnp.conj(b2) * b3 * psi_sfg

    # Row 2: B1^2 Psi(-dk_shg) + 2 B1* B3 Psi(dk_sfg)
    # Note: v2 term 2 uses 2 * kappa_sfg. Psi_sfg includes kappa_sfg. So we just multiply by 2.
    v2 = (b1**2) * psi_neg_shg + 2 * jnp.conj(b1) * b3 * psi_sfg

    # Row 3: 3 B1 B2 Psi(-dk_sfg)
    v3 = 3 * b1 * b2 * psi_neg_sfg

    vector = jnp.array([v1, v2, v3])

    # Multiply by i * exp(L * H)
    # L = diag(0, lin2, lin3)
    exp_lh = jnp.exp(lin * h_total)

    b_nl = 1j * exp_lh * vector

    # Linear evolution of B_0
    b_linear = exp_lh * b_curr

    return b_linear + b_nl


@functools.partial(jax.jit, static_argnames=["return_trace"])
def _super_step_kernel(  # noqa: PLR0913
    widths_padded: jax.Array,
    kappa_shg_padded: jax.Array,
    kappa_sfg_padded: jax.Array,
    delta_k1: jax.Array,
    delta_k2: jax.Array,
    b_initial: jax.Array,
    *,
    return_trace: bool,
) -> tuple[jax.Array, jax.Array | None]:
    lin = _get_lin(delta_k1, delta_k2)

    # scan over blocks
    def scan_body(b_carry: jax.Array, block_data: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array | None]:
        w_block, k_shg_block, k_sfg_block = block_data

        h_total = jnp.sum(w_block)

        psi_factors = _calc_structure_factors(w_block, k_shg_block, k_sfg_block, delta_k1, delta_k2)

        b_next = _super_step_update(b_carry, h_total, psi_factors, lin)

        track_val = b_next if return_trace else None
        return b_next, track_val

    # Group data for scan
    scan_data = (widths_padded, kappa_shg_padded, kappa_sfg_padded)

    b_final, b_stacked = lax.scan(scan_body, b_initial, scan_data)

    if return_trace and b_stacked is not None:
        full_trace = jnp.vstack([b_initial, b_stacked])
        return b_final, full_trace

    return b_final, None


def _validate_inputs(
    domain_widths: jax.Array,
    kappa_shg: jax.Array,
    kappa_sfg: jax.Array,
    b_initial: jax.Array,
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

    if b_initial.shape != (3,) or (b_initial.dtype not in (jnp.complex64, jnp.complex128)):
        msg = "b_initial must be a 1D array of shape (3,) and complex dtype."
        raise ValueError(msg)


def simulate_super_step(  # noqa: PLR0913
    domain_widths: jax.Array,
    kappa_shg_vals: jax.Array,
    kappa_sfg_vals: jax.Array,
    delta_k1: jax.Array,
    delta_k2: jax.Array,
    b_initial: jax.Array,
    block_size: int = 1,
) -> jax.Array:
    """
    Simulates CTHG using the Poly-Domain Analytic Integration (Super-Step) method.

    Args:
        block_size: Number of domains to merge into a single update step.
    """
    _validate_inputs(domain_widths, kappa_shg_vals, kappa_sfg_vals, b_initial)

    n_domains = domain_widths.shape[0]

    # Pad to multiple of block_size
    remainder = n_domains % block_size
    if remainder != 0:
        pad_len = block_size - remainder
        # Pad widths with 0
        widths_new = jnp.pad(domain_widths, (0, pad_len), constant_values=0.0)
        # Pad kappas with 0 (safe as they will be multiplied by phi(w, 0)=0 term roughly, but explicit 0 is clearer)
        k_shg_new = jnp.pad(kappa_shg_vals, (0, pad_len), constant_values=0.0)
        k_sfg_new = jnp.pad(kappa_sfg_vals, (0, pad_len), constant_values=0.0)
    else:
        widths_new = domain_widths
        k_shg_new = kappa_shg_vals
        k_sfg_new = kappa_sfg_vals

    # Reshape
    n_blocks = widths_new.shape[0] // block_size
    w_matrix = widths_new.reshape(n_blocks, block_size)
    k_shg_matrix = k_shg_new.reshape(n_blocks, block_size)
    k_sfg_matrix = k_sfg_new.reshape(n_blocks, block_size)

    b_final, _ = _super_step_kernel(w_matrix, k_shg_matrix, k_sfg_matrix, delta_k1, delta_k2, b_initial, return_trace=False)

    return b_final


def simulate_super_step_with_trace(  # noqa: PLR0913
    domain_widths: jax.Array,
    kappa_shg_vals: jax.Array,
    kappa_sfg_vals: jax.Array,
    delta_k1: jax.Array,
    delta_k2: jax.Array,
    b_initial: jax.Array,
    block_size: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """
    Simulates CTHG using the Poly-Domain Analytic Integration (Super-Step) method.

    Args:
        block_size: Number of domains to merge into a single update step.
    """
    _validate_inputs(domain_widths, kappa_shg_vals, kappa_sfg_vals, b_initial)

    n_domains = domain_widths.shape[0]

    # Pad to multiple of block_size
    remainder = n_domains % block_size
    if remainder != 0:
        pad_len = block_size - remainder
        # Pad widths with 0
        widths_new = jnp.pad(domain_widths, (0, pad_len), constant_values=0.0)
        # Pad kappas with 0 (safe as they will be multiplied by phi(w, 0)=0 term roughly, but explicit 0 is clearer)
        k_shg_new = jnp.pad(kappa_shg_vals, (0, pad_len), constant_values=0.0)
        k_sfg_new = jnp.pad(kappa_sfg_vals, (0, pad_len), constant_values=0.0)
    else:
        widths_new = domain_widths
        k_shg_new = kappa_shg_vals
        k_sfg_new = kappa_sfg_vals

    # Reshape
    n_blocks = widths_new.shape[0] // block_size
    w_matrix = widths_new.reshape(n_blocks, block_size)
    k_shg_matrix = k_shg_new.reshape(n_blocks, block_size)
    k_sfg_matrix = k_sfg_new.reshape(n_blocks, block_size)

    b_final, trace = _super_step_kernel(w_matrix, k_shg_matrix, k_sfg_matrix, delta_k1, delta_k2, b_initial, return_trace=True)

    return b_final, trace
