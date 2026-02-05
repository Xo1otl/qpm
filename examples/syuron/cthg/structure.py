import jax
import jax.numpy as jnp
from dataclasses import dataclass
from qpm import mgoslt

@dataclass
class SimulationConfig:
    shg_len: float = 15000.0
    sfg_len: float = 15000.0
    kappa_shg_base: float = 1.5e-5 / (2 / jnp.pi) * 10
    temperature: float = 70.0
    wavelength: float = 1.064
    input_power: float = 10.0
    block_size: int = round(1445 / 5)
    domain_width_noise_std: float = 0.0
    random_seed: int = 42


@dataclass
class SimulationStructure:
    domain_widths: jax.Array
    kappa_shg_vals: jax.Array
    kappa_sfg_vals: jax.Array
    dk_shg: jax.Array
    dk_sfg: jax.Array
    p_in: jax.Array
    z_coords: jax.Array
    block_size: int


def setup_structure(config: SimulationConfig) -> SimulationStructure:
    kappa_sfg = 2 * config.kappa_shg_base

    dk_shg = mgoslt.calc_twm_delta_k(config.wavelength, config.wavelength, config.temperature)
    dk_sfg = mgoslt.calc_twm_delta_k(config.wavelength, config.wavelength / 2, config.temperature)

    lc_shg = jnp.abs(jnp.pi / dk_shg)
    lc_sfg = jnp.abs(jnp.pi / dk_sfg)

    n_shg = int(config.shg_len / lc_shg)
    widths_shg = jnp.full(n_shg, lc_shg)

    n_sfg = int(config.sfg_len / lc_sfg)
    widths_sfg = jnp.full(n_sfg, lc_sfg)

    domain_widths = jnp.concatenate([widths_shg, widths_sfg])
    
    # Apply noise if specified
    if config.domain_width_noise_std > 0.0:
        key = jax.random.PRNGKey(config.random_seed)
        noise = jax.random.normal(key, domain_widths.shape) * config.domain_width_noise_std
        domain_widths = domain_widths + noise
        # Ensure widths remain positive? Usually noise is small (microns vs 15um LC), but good hygiene.
        domain_widths = jnp.maximum(domain_widths, 0.0)

    num_domains = len(domain_widths)

    sign_pattern = jnp.array([1.0 if i % 2 == 0 else -1.0 for i in range(num_domains)])
    kappa_shg_vals = config.kappa_shg_base * sign_pattern
    kappa_sfg_vals = kappa_sfg * sign_pattern

    p_in = jnp.array([jnp.sqrt(config.input_power), 0.0, 0.0], dtype=jnp.complex128)
    z_coords = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(domain_widths)])

    return SimulationStructure(
        domain_widths=domain_widths,
        kappa_shg_vals=kappa_shg_vals,
        kappa_sfg_vals=kappa_sfg_vals,
        dk_shg=dk_shg,
        dk_sfg=dk_sfg,
        p_in=p_in,
        z_coords=z_coords,
        block_size=config.block_size,
    )
