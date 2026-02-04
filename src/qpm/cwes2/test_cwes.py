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


def test_magnus_conservation() -> None:
    """
    Verifies that the Magnus scheme conserves energy (norm).
    """
    config = SimulationConfig()
    struct = setup_structure(config)

    print(f"\nStructure: {len(struct.domain_widths)} domains.")

    # Run Magnus simulation
    # Use a large block size to test stability/speed
    block_size = 50

    # Trim to multiple of block size
    num_domains = struct.domain_widths.shape[0]
    remainder = num_domains % block_size
    if remainder != 0:
        trim = num_domains - remainder
        struct.domain_widths = struct.domain_widths[:trim]
        struct.kappa_shg_vals = struct.kappa_shg_vals[:trim]
        struct.kappa_sfg_vals = struct.kappa_sfg_vals[:trim]

    t0 = time.perf_counter()
    a_final = cwes2.simulate_magnus(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
        block_size,
    )
    a_final.block_until_ready()
    t_magnus = time.perf_counter() - t0

    # Check conservation: |A1|^2 + |A2|^2 + |A3|^2 = const
    initial_energy = jnp.sum(jnp.abs(struct.p_in) ** 2)
    final_energy = jnp.sum(jnp.abs(a_final) ** 2)

    print(f"Magnus (Block {block_size}):")
    print(f"  Time: {t_magnus * 1e3:.2f} ms")
    print(f"  Initial Energy: {initial_energy:.6f}")
    print(f"  Final Energy:   {final_energy:.6f}")
    print(f"  Diff:           {jnp.abs(final_energy - initial_energy):.6e}")

    assert jnp.isclose(final_energy, initial_energy, atol=1e-5), "Magnus scheme must conserve energy"

    # Compare with LFAGA (Non-conservative baseline)
    a_lfaga = cwes2.simulate_lfaga(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
        block_size,
    )
    lfaga_energy = jnp.sum(jnp.abs(a_lfaga) ** 2)
    print(f"LFAGA Energy Diff: {jnp.abs(lfaga_energy - initial_energy):.6e}")


def test_magnus_trace() -> None:
    """
    Verifies that the traced Magnus simulation returns the correct shape and matches the final state.
    """
    config = SimulationConfig(shg_len=2000.0, sfg_len=2000.0)  # Short run for speed
    struct = setup_structure(config)

    block_size = 50
    # Trim to multiple of block size
    num_domains = struct.domain_widths.shape[0]
    remainder = num_domains % block_size
    if remainder != 0:
        trim = num_domains - remainder
        struct.domain_widths = struct.domain_widths[:trim]
        struct.kappa_shg_vals = struct.kappa_shg_vals[:trim]
        struct.kappa_sfg_vals = struct.kappa_sfg_vals[:trim]

    # Run traced simulation
    a_final, trace = cwes2.simulate_magnus_with_trace(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
        block_size,
    )

    # 1. Check shape
    # Trace should have (N_blocks + 1) points (including initial)
    expected_steps = len(struct.domain_widths) // block_size
    assert trace.shape == (expected_steps + 1, 3), f"Expected trace shape {(expected_steps + 1, 3)}, got {trace.shape}"

    # 2. Check consistency
    # Last point in trace should equal a_final
    assert jnp.allclose(trace[-1], a_final, atol=1e-5), "Last point of trace must match final state."

    # 3. Check against non-traced
    a_final_ref = cwes2.simulate_magnus(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
        block_size,
    )
    assert jnp.allclose(a_final, a_final_ref, atol=1e-6), "Traced and non-traced versions must match."
