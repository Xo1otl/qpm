import jax
import jax.numpy as jnp


def _r(a: jax.Array, b: jax.Array, k: jax.Array) -> jax.Array:
    ell = b - a
    center = (a + b) / 2.0
    sinc_arg = k * ell / (2.0 * jnp.pi)
    sinc_val = jnp.sinc(sinc_arg)

    return jnp.exp(-1j * k * center) * ell * sinc_val


def _j(a: jax.Array, b: jax.Array, dk1: jax.Array, dk2: jax.Array) -> jax.Array:
    ell = b - a
    dk1_safe = jnp.where(dk1 == 0.0, 1.0, dk1)
    dk2_safe = jnp.where(dk2 == 0.0, 1.0, dk2)
    r_dk2 = _r(a, b, dk2)
    r_dk1_dk2 = _r(a, b, dk1 + dk2)
    j_general = (jnp.exp(-1j * dk1 * a) * r_dk2 - r_dk1_dk2) / (1j * dk1_safe)
    j_dk1_zero_dk2_zero = ell**2 / 2.0
    term1 = 1j * ell * jnp.exp(-1j * dk2 * b) / dk2_safe
    term2 = (jnp.exp(-1j * dk2 * b) - jnp.exp(-1j * dk2 * a)) / (dk2_safe**2)
    j_dk1_zero_dk2_nonzero = term1 + term2
    j_dk1_zero = jnp.where(dk2 == 0, j_dk1_zero_dk2_zero, j_dk1_zero_dk2_nonzero)
    return jnp.where(dk1 == 0, j_dk1_zero, j_general)


@jax.jit
def calc_a3_npda(kappas_shg: jax.Array, kappas_sfg: jax.Array, widths: jax.Array, dk1: jax.Array, dk2: jax.Array) -> jax.Array:
    z_end = jnp.cumsum(widths)
    z_start = jnp.pad(z_end[:-1], (1, 0))
    j_n = _j(z_start, z_end, dk1, dk2)
    diagonal_sum = jnp.sum(kappas_shg * kappas_sfg * j_n)
    s1_n = kappas_shg * _r(z_start, z_end, dk1)
    s2_n = kappas_sfg * _r(z_start, z_end, dk2)
    inner_sum_cumulative = jnp.cumsum(s1_n)
    inner_sum_shifted = jnp.pad(inner_sum_cumulative, (1, 0))[:-1]
    double_sum = jnp.sum(s2_n * inner_sum_shifted)
    return double_sum + diagonal_sum
