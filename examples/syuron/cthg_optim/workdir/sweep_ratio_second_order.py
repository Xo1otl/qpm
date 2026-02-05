import pickle
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from qpm import cwes2, mgoslt

# Enable x64
jax.config.update("jax_enable_x64", val=True)


def build_simulation_kernel(wavelength: float, temperature: float, total_length: float) -> tuple[Callable, int, int, float]:
    # --- 1. Physics Calculations ---
    dk1 = float(mgoslt.calc_twm_delta_k(wavelength, wavelength, temperature))
    dk2 = float(mgoslt.calc_twm_delta_k(wavelength, wavelength / 2, temperature))

    lc1 = np.pi / dk1
    width_shg = lc1
    period_shg = 2 * width_shg

    # We estimate min width to determine buffer size
    min_possible_width = 1.5
    max_n_domains = int(np.ceil(total_length / min_possible_width))
    max_shg_cycles = int(total_length / period_shg)

    # --- 2. Constant Constants ---
    k_val = 1.5e-5 / (2 / np.pi)
    k_shg_val = k_val
    k_sfg_val = 2 * k_val

    b_init = jnp.array([np.sqrt(10.0), 0.0, 0.0], dtype=jnp.complex128)

    # Poling signs: [1, -1, 1, -1, ...]
    signs = jnp.tile(jnp.array([1.0, -1.0]), (max_n_domains // 2 + 1))[:max_n_domains]
    k_shg_arr = signs * k_shg_val
    k_sfg_arr = signs * k_sfg_val

    # --- 3. JIT Kernels ---
    # Kernel for sweeping: returns ONLY amplitude to save memory
    @jax.jit
    def simulate_amp_only(params):
        # params: [n_shg_cycles, sfg_w1, sfg_w2]
        n_shg_cycles = params[0]
        w1 = params[1]
        w2 = params[2]

        # Geometry
        shg_len = n_shg_cycles * period_shg
        n_shg_domains = (n_shg_cycles * 2).astype(int)

        rem_len = total_length - shg_len
        sfg_period = w1 + w2
        # Number of full SFG periods fitting in remaining length
        n_sfg_periods = jnp.maximum(0, jnp.floor(rem_len / sfg_period).astype(int))
        n_sfg_domains = n_sfg_periods * 2

        # Construction
        idx = jnp.arange(max_n_domains)

        # Masks
        is_shg = idx < n_shg_domains
        is_sfg_region = (idx >= n_shg_domains) & (idx < (n_shg_domains + n_sfg_domains))

        # For SFG, we need alternating w1, w2
        sfg_rel_idx = idx - n_shg_domains
        is_w1 = is_sfg_region & (sfg_rel_idx % 2 == 0)
        is_w2 = is_sfg_region & (sfg_rel_idx % 2 == 1)

        # Compose widths
        w = jnp.zeros(max_n_domains, dtype=jnp.float64)
        w = jnp.where(is_shg, width_shg, w)
        w = jnp.where(is_w1, w1, w)
        w = jnp.where(is_w2, w2, w)

        # Simulate
        b_res = cwes2.simulate_twm(w, k_shg_arr, k_sfg_arr, dk1, dk2, b_init)

        return jnp.abs(b_res[2])

    # Helper to reconstruct widths for the best result
    @jax.jit
    def get_structure(params):
        n_shg_cycles = params[0]
        w1 = params[1]
        w2 = params[2]

        shg_len = n_shg_cycles * period_shg
        n_shg_domains = (n_shg_cycles * 2).astype(int)

        rem_len = total_length - shg_len
        sfg_period = w1 + w2
        n_sfg_periods = jnp.maximum(0, jnp.floor(rem_len / sfg_period).astype(int))
        n_sfg_domains = n_sfg_periods * 2

        idx = jnp.arange(max_n_domains)
        is_shg = idx < n_shg_domains
        is_sfg_region = (idx >= n_shg_domains) & (idx < (n_shg_domains + n_sfg_domains))
        sfg_rel_idx = idx - n_shg_domains
        is_w1 = is_sfg_region & (sfg_rel_idx % 2 == 0)
        is_w2 = is_sfg_region & (sfg_rel_idx % 2 == 1)

        w = jnp.zeros(max_n_domains, dtype=jnp.float64)
        w = jnp.where(is_shg, width_shg, w)
        w = jnp.where(is_w1, w1, w)
        w = jnp.where(is_w2, w2, w)
        return w

    return simulate_amp_only, get_structure, max_shg_cycles, max_n_domains, width_shg


def main():
    print("Optimization: Second-Order QPM Initialization Search")

    # Config
    params = {
        "wavelength": 1.064,
        "temperature": 70.0,
        "total_length": 17500.0,
    }

    sim_kernel, struct_kernel, max_cyc, max_n, shg_w = build_simulation_kernel(**params)
    print(f"Max SHG Cycles: {max_cyc}, Buffer Size: {max_n}")

    # --- Theoretical Width Determination ---
    dk2 = float(mgoslt.calc_twm_delta_k(params["wavelength"], params["wavelength"] / 2, params["temperature"]))
    lc2 = np.pi / dk2
    target_period = 4 * lc2  # Second-order QPM period
    print(f"Lc2: {lc2:.4f}, Target Period (2nd order): {target_period:.4f}")

    # Theoretical Optima Analysis:
    # 2nd order efficiency ~ sin(2*pi*D). Max at D=0.25 (w1=Lc2=1.1um).
    # Constraint: w >= 1.5. This forces us away from D=0.25.
    # We should pick the valid w1 closest to 1.1um, which is exactly the limit 1.5um.
    w1_fixed = 1.5
    w2_fixed = target_period - w1_fixed

    print(f"Theoretical Constraints: Fixed w1={w1_fixed:.4f}, w2={w2_fixed:.4f} (Duty={w1_fixed / target_period:.2f})")

    # Sweep SHG Cycles: 0 to Max
    # 1D Sweep is fast enough to do all at once (~2200 points)
    cycles_grid = jnp.arange(0, max_cyc + 1, dtype=jnp.float64)
    n_configs = len(cycles_grid)

    # Broadcast params for 1D input
    # cycles_grid is (N,), we need (N, 1, 1) or similar if we reuse the kernel?
    # No, kernel expects params=[C, w1, w2].

    # Create (N, 3) array
    w1_col = jnp.full((n_configs,), w1_fixed)
    w2_col = jnp.full((n_configs,), w2_fixed)

    batch_params = jnp.stack([cycles_grid, w1_col, w2_col], axis=1)

    print(f"Testing {n_configs} configurations (1D Sweep)...")
    t0 = time.time()

    # Run Batched Simulation (Single Batch is fine for 2k items)
    amps = jax.vmap(sim_kernel)(batch_params)
    amps = jax.block_until_ready(amps)

    elapsed = time.time() - t0

    # Find Best
    best_idx = jnp.argmax(amps)
    best_amp = float(amps[best_idx])
    best_p = batch_params[best_idx]

    print(f"Total time: {elapsed:.2f}s")
    print(f"Best Amp: {best_amp:.6f} @ C={best_p[0]:.0f}, w1={best_p[1]:.3f}, w2={best_p[2]:.3f}")

    # Reconstruct Structure
    best_w = struct_kernel(best_p)
    w_clean = best_w[best_w > 0]

    # Save
    data = {
        "widths": np.array(w_clean),
        "amp": best_amp,
        "params": params,
        "sweep_params": {"cycles": int(best_p[0]), "w1": float(best_p[1]), "w2": float(best_p[2])},
    }

    out_file = "best_2nd_order_init.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
