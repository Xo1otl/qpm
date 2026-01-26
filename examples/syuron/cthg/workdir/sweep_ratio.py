import pickle
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from qpm import cwes2, mgoslt

# Enable x64 at module level
jax.config.update("jax_enable_x64", val=True)


def build_simulation_kernel(wavelength: float, temperature: float, total_length: float) -> tuple[Callable, int, int]:
    """
    Factory that pre-calculates physics constants and returns a JIT-compiled
    simulation function tailored to the specific geometry.
    """

    # --- 1. Physics Calculations ---
    # Calculate wavevector mismatches
    dk1 = float(mgoslt.calc_twm_delta_k(wavelength, wavelength, temperature))
    dk2 = float(mgoslt.calc_twm_delta_k(wavelength, wavelength / 2, temperature))

    # Calculate Coherence Lengths (Domain Widths)
    lc1 = np.pi / dk1  # SHG domain width
    lc2 = np.pi / dk2  # Base SFG parameter

    # Define Periodicity
    # SHG: period is 2 * lc1
    # SFG: defined in original code as width=3*lc2, period=2*width
    width_shg = lc1
    width_sfg = 3 * lc2

    period_shg = 2 * width_shg
    period_sfg = 2 * width_sfg

    # --- 2. Dynamic Array Sizing ---
    # Determine the maximum array size (MAX_N) needed for the buffer.
    # We divide total length by the smallest possible domain width.
    min_domain_width = min(width_shg, width_sfg)
    max_n_domains = int(np.ceil(total_length / min_domain_width))

    # Calculate sweep limits
    max_shg_cycles = int(total_length / period_shg)

    # --- 3. Constant Tensor Prep ---
    # Pre-calculate coupling constants to avoid re-computing in the loop
    # d_eff ~ 1.5e-5 (assumed from original snippet)
    k_val_shg = 1.5e-5 / (2 / np.pi)
    k_val_sfg = 1.5e-5 / (2 / np.pi) * 2

    amp_fund = np.sqrt(10.0)
    b_init = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)

    # Pre-compute sign alteration array for MAX_N
    # [1, -1, 1, -1, ...]
    signs = jnp.tile(jnp.array([1.0, -1.0]), (max_n_domains // 2 + 1))[:max_n_domains]
    k_shg_arr = signs * k_val_shg
    k_sfg_arr = signs * k_val_sfg

    # --- 4. Closure Definition ---
    @jax.jit
    def compute_amp_for_cycles(n_shg_cycles):
        """
        Simulates the structure for a specific number of SHG cycles.
        """
        # --- Structure Generation ---
        actual_shg_len = n_shg_cycles * period_shg
        n_shg_domains = n_shg_cycles * 2

        # Calculate remaining length for SFG
        rem_len = total_length - actual_shg_len
        n_sfg_cycles = jnp.maximum(0, jnp.floor(rem_len / period_sfg).astype(int))
        n_sfg_domains = n_sfg_cycles * 2

        # Create geometric masks
        idx = jnp.arange(max_n_domains)
        is_shg = idx < n_shg_domains
        is_sfg = (idx >= n_shg_domains) & (idx < (n_shg_domains + n_sfg_domains))

        # Assign domain widths based on masks (vectorized)
        # Default is 0.0 (padding)
        w = jnp.zeros(max_n_domains, dtype=jnp.float64)
        w = jnp.where(is_shg, width_shg, w)
        w = jnp.where(is_sfg, width_sfg, w)

        # --- Simulation ---
        # Note: cwes2.simulate_twm must handle padded (0 width) domains gracefully
        # or we must assume they don't contribute to evolution.
        b_res = cwes2.simulate_twm(w, k_shg_arr, k_sfg_arr, dk1, dk2, b_init)

        return jnp.abs(b_res[2]), w

    return compute_amp_for_cycles, max_shg_cycles, max_n_domains


def main():
    print("Vectorized Seed Tuning (Full Cycle Search, JAX)...")

    # Configuration
    params = {
        "wavelength": 1.064,
        "temperature": 70.0,
        "total_length": 17500.0,
    }

    # 1. Build the kernel
    sim_kernel, max_shg_cycles, buffer_size = build_simulation_kernel(**params)

    print(f"Structure Buffer Size (MAX_N): {buffer_size}")
    print(f"Sweeping 0 to {max_shg_cycles} SHG cycles...")

    # 2. Vectorize over the sweep range
    cycles_arr = jnp.arange(max_shg_cycles + 1)

    t0 = time.time()
    # Batch the computation
    amps, widths = jax.vmap(sim_kernel)(cycles_arr)

    # Force synchronization for timing
    amps.block_until_ready()
    elapsed = time.time() - t0
    print(f"Simulation took: {elapsed:.2f}s")

    # 3. Analyze Results
    best_idx = jnp.argmax(amps)
    best_amp = float(amps[best_idx])
    best_w = widths[best_idx]
    best_cycles = int(cycles_arr[best_idx])

    print(f"Best Amp: {best_amp:.6f}")
    print(f"Best Params: SHG Cycles {best_cycles}")

    # 4. Save
    # Remove padding (zeros) before saving
    w_clean = best_w[best_w > 0]

    output_data = {"widths": np.array(w_clean), "amp": best_amp, "shg_cycles": best_cycles, "params": params}

    with open("best_ratio.pkl", "wb") as f:
        pickle.dump(output_data, f)
    print("Saved results to best_ratio.pkl")


if __name__ == "__main__":
    main()
