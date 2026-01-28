import jax
import jax.numpy as jnp


def calculate_local_shg_amplitudes(
    domain_widths: jax.Array,
    kappa_vals: jax.Array,
    delta_k: jax.Array,
    b_initial: jax.Array,
) -> jax.Array:
    gamma = delta_k / 2.0
    a_omega_sq = b_initial**2
    gamma_l = gamma * domain_widths
    sinc_term = jnp.sinc(gamma_l / jnp.pi)
    return -1j * kappa_vals * a_omega_sq * domain_widths * jnp.exp(1j * gamma_l) * sinc_term


def simulate_shg_npda(
    domain_widths: jax.Array,
    kappa_vals: jax.Array,
    delta_k: jax.Array,
    b_initial: jax.Array,
) -> jax.Array:
    local_amplitudes = calculate_local_shg_amplitudes(domain_widths, kappa_vals, delta_k, b_initial)
    z_starts = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(domain_widths[:-1])])
    phase_factors = jnp.exp(1j * delta_k * z_starts)
    return jnp.sum(local_amplitudes * phase_factors)


def simulate_shg_npda_trace(
    domain_widths: jax.Array,
    kappa_vals: jax.Array,
    delta_k: jax.Array,
    b_initial: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    local_amplitudes = calculate_local_shg_amplitudes(domain_widths, kappa_vals, delta_k, b_initial)
    z_starts = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(domain_widths[:-1])])
    phase_factors = jnp.exp(1j * delta_k * z_starts)
    terms_to_sum = local_amplitudes * phase_factors
    cumulative_amplitudes = jnp.cumsum(terms_to_sum)
    shg_amplitude_trace = jnp.concatenate([jnp.array([0.0j]), cumulative_amplitudes])
    z_coords = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(domain_widths)])
    return z_coords, shg_amplitude_trace
