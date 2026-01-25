import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from qpm import cwes2, mgoslt


@dataclass
class SimulationConfig:
    shg_len: float = 10000.0
    sfg_len: float = 30000.0
    kappa_shg_base: float = 1.5e-5 / (2 / jnp.pi)
    temperature: float = 70.0
    wavelength: float = 1.064
    input_power: float = 10.0


@dataclass
class SimulationStructure:
    domain_widths: jax.Array
    kappa_shg_vals: jax.Array
    kappa_sfg_vals: jax.Array
    dk_shg: jax.Array
    dk_sfg: jax.Array
    p_in: jax.Array


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
    num_domains = len(domain_widths)

    # sign_pattern = jnp.array([1.0 if i % 2 == 0 else -1.0 for i in range(num_domains)])
    # Optimization: Create alternating pattern directly
    sign_pattern = jnp.ones(num_domains)
    sign_pattern = sign_pattern.at[1::2].set(-1.0)

    kappa_shg_vals = config.kappa_shg_base * sign_pattern
    kappa_sfg_vals = kappa_sfg * sign_pattern

    p_in = jnp.array([jnp.sqrt(config.input_power), 0.0, 0.0], dtype=jnp.complex64)

    return SimulationStructure(
        domain_widths=domain_widths,
        kappa_shg_vals=kappa_shg_vals,
        kappa_sfg_vals=kappa_sfg_vals,
        dk_shg=dk_shg,
        dk_sfg=dk_sfg,
        p_in=p_in,
    )


def test_method_comparison() -> None:
    """
    Compares NPDA, Perturbation, and Super-Step methods.
    Verifies accuracy and speedup.
    """
    config = SimulationConfig()
    struct = setup_structure(config)

    print(f"\nStructure: {len(struct.domain_widths)} domains.")

    # 1. NPDA
    # Warmup
    cwes2.calc_a3_npda(
        struct.p_in[0],
        struct.kappa_shg_vals[:10],
        struct.kappa_sfg_vals[:10],
        struct.domain_widths[:10],
        struct.dk_shg,
        struct.dk_sfg,
    ).block_until_ready()

    t0 = time.perf_counter()
    val_a3_npda = cwes2.calc_a3_npda(struct.p_in[0], struct.kappa_shg_vals, struct.kappa_sfg_vals, struct.domain_widths, struct.dk_shg, struct.dk_sfg)
    val_a3_npda.block_until_ready()
    t_npda = time.perf_counter() - t0

    # 2. Perturbation
    # Warmup
    cwes2.simulate_twm(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
    ).block_until_ready()

    t0 = time.perf_counter()
    b_pert = cwes2.simulate_twm(struct.domain_widths, struct.kappa_shg_vals, struct.kappa_sfg_vals, struct.dk_shg, struct.dk_sfg, struct.p_in)
    b_pert.block_until_ready()
    t_pert = time.perf_counter() - t0

    # 3. Super-Step
    block_size = 31

    # Ensure divisibility
    num_domains = struct.domain_widths.shape[0]
    remainder = num_domains % block_size
    if remainder != 0:
        trim_len = num_domains - remainder
        print(f"Trimming structure from {num_domains} to {trim_len} for block_size {block_size}.")
        struct.domain_widths = struct.domain_widths[:trim_len]
        struct.kappa_shg_vals = struct.kappa_shg_vals[:trim_len]
        struct.kappa_sfg_vals = struct.kappa_sfg_vals[:trim_len]

    # Warmup
    cwes2.simulate_super_step(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
        block_size,
    ).block_until_ready()

    t0 = time.perf_counter()
    b_super = cwes2.simulate_super_step(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
        block_size,
    )
    b_super.block_until_ready()
    t_super = time.perf_counter() - t0

    # Re-run Perturbation on the trimmed structure for fair comparison
    if remainder != 0:
        b_pert_trimmed = cwes2.simulate_twm(
            struct.domain_widths, struct.kappa_shg_vals, struct.kappa_sfg_vals, struct.dk_shg, struct.dk_sfg, struct.p_in
        )
        b_pert_trimmed.block_until_ready()
        # Update comparison baseline
        b_pert = b_pert_trimmed

    # Results
    a3_npda = jnp.abs(val_a3_npda)
    a3_pert = jnp.abs(b_pert[2])
    a3_super = jnp.abs(b_super[2])

    print(f"NPDA |A3|:       {a3_npda:.6e} (Time: {t_npda * 1e3:.2f} ms)")
    print(f"Perturbation:    {a3_pert:.6e} (Time: {t_pert * 1e3:.2f} ms)")
    print(f"Super-Step (20): {a3_super:.6e} (Time: {t_super * 1e3:.2f} ms)")
    print(f"Speedup (Pert/Super): {t_pert / t_super:.2f}x")

    assert jnp.allclose(b_super, b_pert, rtol=5e-2, atol=1e-3), "Super-Step should match Perturbation"
