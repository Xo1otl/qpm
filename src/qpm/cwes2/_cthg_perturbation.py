import functools

import jax
import jax.numpy as jnp
from jax import lax

OMEGA_SMALL_EPS: float = 1e-9


def _get_lin(delta_k1: jax.Array, delta_k2: jax.Array) -> jax.Array:
    return jnp.array([0.0, delta_k1 * 1j, (delta_k1 + delta_k2) * 1j], dtype=jnp.complex64)


def _phi_optimized(exp_val: jax.Array, omega: jax.Array, h: jax.Array) -> jax.Array:
    val_small = h + h**2 * omega / 2.0
    val_large = (exp_val - 1.0) / omega
    return jnp.where(jnp.abs(omega) < OMEGA_SMALL_EPS, val_small, val_large)


def _propagate_step(b_in: jax.Array, h: jax.Array, kappa_shg: jax.Array, kappa_sfg: jax.Array, lin: jax.Array) -> jax.Array:
    b1n, b2n, b3n = b_in
    _, lin2, lin3 = lin

    exp_linh = jnp.exp(lin * h)
    e1, e2, e3 = exp_linh

    omega_a = lin2
    omega_b = lin3 - lin2

    exp_omega_a = e2
    exp_omega_b = e3 * jnp.conj(e2)

    phi_a = _phi_optimized(exp_omega_a, omega_a, h)
    phi_b = _phi_optimized(exp_omega_b, omega_b, h)
    phi_neg_a = _phi_optimized(jnp.conj(exp_omega_a), -omega_a, h)
    phi_neg_b = _phi_optimized(jnp.conj(exp_omega_b), -omega_b, h)

    delta_b_nlin1 = 1j * e1 * (kappa_shg * jnp.conj(b1n) * b2n * phi_a + kappa_sfg * jnp.conj(b2n) * b3n * phi_b)
    delta_b_nlin2 = 1j * e2 * (kappa_shg * b1n**2 * phi_neg_a + 2 * kappa_sfg * jnp.conj(b1n) * b3n * phi_b)
    delta_b_nlin3 = 1j * 3 * kappa_sfg * e3 * (b1n * b2n * phi_neg_b)

    return exp_linh * b_in + jnp.array([delta_b_nlin1, delta_b_nlin2, delta_b_nlin3])


@functools.partial(jax.jit, static_argnames=["return_trace"])
def _twm_kernel(  # noqa: PLR0913
    domain_widths: jax.Array,
    kappa_shg_vals: jax.Array,
    kappa_sfg_vals: jax.Array,
    delta_k1: jax.Array,
    delta_k2: jax.Array,
    b_initial: jax.Array,
    *,
    return_trace: bool,
) -> tuple[jax.Array, jax.Array | None]:
    lin = _get_lin(delta_k1, delta_k2)

    def scan_body(b_carry: jax.Array, domain_params: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array | None]:
        h, k_shg, k_sfg = domain_params
        b_next = _propagate_step(b_carry, h, k_shg, k_sfg, lin)
        track_val = b_next if return_trace else None
        return b_next, track_val

    b_final, b_stacked = lax.scan(scan_body, b_initial, (domain_widths, kappa_shg_vals, kappa_sfg_vals))

    if return_trace and b_stacked is not None:
        full_trace = jnp.vstack([b_initial, b_stacked])
        return b_final, full_trace

    return b_final, None


def _validate_inputs(
    domain_widths: jax.Array,
    kappa_shg_vals: jax.Array,
    kappa_sfg_vals: jax.Array,
    b_initial: jax.Array,
) -> None:
    if (
        domain_widths.ndim != 1
        or kappa_shg_vals.ndim != 1
        or kappa_sfg_vals.ndim != 1
        or domain_widths.shape != kappa_shg_vals.shape
        or domain_widths.shape != kappa_sfg_vals.shape
    ):
        msg = "domain_widths and kappa_vals must be 1D arrays of the same shape."
        raise ValueError(msg)

    if b_initial.shape != (3,) or (b_initial.dtype not in (jnp.complex64, jnp.complex128)):
        msg = "b_initial must be a 1D array of shape (3,) and complex dtype."
        raise ValueError(msg)


def simulate_twm(  # noqa: PLR0913
    domain_widths: jax.Array,
    kappa_shg_vals: jax.Array,
    kappa_sfg_vals: jax.Array,
    delta_k1: jax.Array,
    delta_k2: jax.Array,
    b_initial: jax.Array,
) -> jax.Array:
    _validate_inputs(domain_widths, kappa_shg_vals, kappa_sfg_vals, b_initial)
    b_final, _ = _twm_kernel(domain_widths, kappa_shg_vals, kappa_sfg_vals, delta_k1, delta_k2, b_initial, return_trace=False)
    return b_final


def simulate_twm_with_trace(  # noqa: PLR0913
    domain_widths: jax.Array,
    kappa_shg_vals: jax.Array,
    kappa_sfg_vals: jax.Array,
    delta_k1: jax.Array,
    delta_k2: jax.Array,
    b_initial: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    _validate_inputs(domain_widths, kappa_shg_vals, kappa_sfg_vals, b_initial)
    b_final, trace = _twm_kernel(domain_widths, kappa_shg_vals, kappa_sfg_vals, delta_k1, delta_k2, b_initial, return_trace=True)
    return b_final, trace
