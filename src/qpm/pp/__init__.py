import jax.numpy as jnp


def generate_section(length: float, period: float, kappa_mag: float) -> jnp.ndarray:
    """指定された長さ、周期、結合係数でQPMセクションを生成する"""
    domain_len = period / 2.0
    num_domains = jnp.floor(length / domain_len).astype(jnp.int32)
    indices = jnp.arange(num_domains)
    h_values = jnp.ones_like(indices, dtype=jnp.float32) * domain_len
    alternating_signs = jnp.power(-1.0, indices)
    kappa_values = kappa_mag * alternating_signs
    return jnp.stack([h_values, kappa_values], axis=1)
