from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap

from qpm import cwes2, mgoslt


@dataclass
class SimulationConfig:
    device_len_um: float = 3000.0

    # Normalized efficiencies approx
    kappa_shg: float = 1.5e-5 * (jnp.pi / 2)
    kappa_sfg: float = 1.5e-5 * (jnp.pi / 2) * 2

    temperature: float = 70.0
    wavelength_um: float = 1.064
    input_power: float = 10.0

    # Grid search
    scan_points: int = 100


def run() -> None:
    # 1. Setup Constants
    config = SimulationConfig()

    lam_fun = config.wavelength_um
    lam_shg = lam_fun / 2.0

    # Calculate mismatch
    # SHG: w + w -> 2w
    dk_shg = mgoslt.calc_twm_delta_k(lam_fun, lam_fun, config.temperature)
    # SFG: w + 2w -> 3w
    dk_sfg = mgoslt.calc_twm_delta_k(lam_fun, lam_shg, config.temperature)

    # Coherence lengths
    lc_shg = jnp.abs(jnp.pi / dk_shg)
    lc_sfg = jnp.abs(jnp.pi / dk_sfg)  # should be smaller

    print(f"Lc_SHG: {lc_shg:.4f} um")
    print(f"Lc_SFG: {lc_sfg:.4f} um")

    # 2. Determine Max Domains (Full SFG filling)
    n_max = int(jnp.ceil(config.device_len_um / lc_sfg))
    print(f"Max Domains (N_max): {n_max}")

    # 3. Generate batch of structures
    ratios = jnp.linspace(0, 1, config.scan_points)

    def make_structure(r: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        l_target_shg = r * config.device_len_um
        l_target_sfg = config.device_len_um - l_target_shg

        n_shg = jnp.round(l_target_shg / lc_shg).astype(jnp.int32)
        n_sfg = jnp.round(l_target_sfg / lc_sfg).astype(jnp.int32)

        # Create indices
        idx = jnp.arange(n_max)

        # Masks
        mask_shg = idx < n_shg
        mask_sfg = (idx >= n_shg) & (idx < n_shg + n_sfg)

        # Widths
        w = jnp.where(mask_shg, lc_shg, jnp.where(mask_sfg, lc_sfg, 0.0))

        # Signs (alternating + - + - ...)
        # SHG: 0 start -> 1, -1, 1...
        sign_shg = 1.0 - 2.0 * (idx % 2)
        # SFG: starts at n_shg. We want n_shg to be +, n_shg+1 to be -, etc.
        # So effective index is (idx - n_shg)
        sign_sfg = 1.0 - 2.0 * ((idx - n_shg) % 2)

        # Kappas
        k1 = jnp.where(mask_shg, config.kappa_shg * sign_shg, 0.0)
        k2 = jnp.where(mask_sfg, config.kappa_sfg * sign_sfg, 0.0)

        # Actual ratio calculation
        # l_curr = jnp.sum(w)
        # ratio = l_shg / l_curr
        l_shg_actual = jnp.sum(jnp.where(mask_shg, lc_shg, 0.0))
        l_curr_actual = jnp.sum(w)

        # Avoid division by zero
        r_actual = jnp.where(l_curr_actual > 0, l_shg_actual / l_curr_actual, 0.0)

        return w, k1, k2, r_actual

    print("Generating structures with JAX...")
    width_stack, k_shg_stack, k_sfg_stack, actual_ratios = vmap(make_structure)(ratios)

    print(f"Batch shape: {width_stack.shape}")

    # 4. Factory for vmap
    block_k = 24

    # The factory:
    def make_simulator() -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
        # Capture constants
        dk1 = dk_shg
        dk2 = dk_sfg
        b0 = jnp.array([jnp.sqrt(config.input_power), 0.0, 0.0], dtype=jnp.complex64)

        def sim_func(w: jax.Array, k1: jax.Array, k2: jax.Array) -> jax.Array:
            return cwes2.simulate_super_step(
                domain_widths=w,
                kappa_shg_vals=k1,
                kappa_sfg_vals=k2,
                delta_k1=dk1,
                delta_k2=dk2,
                b_initial=b0,
                block_size=block_k,
            )

        return jax.jit(vmap(sim_func))

    simulator = make_simulator()

    # Run
    print("Running simulation...")
    b_out_batch = simulator(width_stack, k_shg_stack, k_sfg_stack)

    # Extract THG amplitude (A3?)
    b3_abs = jnp.abs(b_out_batch[:, 2])

    # 5. Plot
    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(10, 6))
    plt.plot(actual_ratios, b3_abs, "o-", label="THW Amp")
    plt.title("Tandem Structure Optimization (3mm)", fontsize=18)
    plt.xlabel("SHG Region Length Ratio", fontsize=16)
    plt.ylabel("THW Amplitude |B3|", fontsize=16)
    plt.tick_params(labelsize=14)
    plt.grid(visible=True)
    plt.tight_layout()
    plt.savefig("thg_optimization.png")
    print("Saved plot to thg_optimization.png")

    # Verify optimal
    max_idx = jnp.argmax(b3_abs)
    best_ratio = actual_ratios[max_idx]
    print(f"Optimal Ratio: {best_ratio:.3f}")
    print(f"Max Amplitude: {b3_abs[max_idx]:.4e}")


if __name__ == "__main__":
    run()
