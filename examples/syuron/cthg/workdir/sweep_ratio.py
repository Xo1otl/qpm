import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from qpm import cwes2, mgoslt

jax.config.update("jax_enable_x64", val=True)

# Constants
wavelength = 1.064
temperature = 70.0
TOTAL_LENGTH = 20000.0
MIN_WIDTH = 1.5
MAX_N = 25000


def main():
    print("Vectorized Seed Tuning (Full Cycle Search, JAX)...")

    dk1 = float(mgoslt.calc_twm_delta_k(wavelength, wavelength, temperature))
    dk2 = float(mgoslt.calc_twm_delta_k(wavelength, wavelength / 2, temperature))

    k_val_shg = 1.5e-5 / (2 / np.pi)
    k_val_sfg = 1.5e-5 / (2 / np.pi) * 2
    amp_fund = jnp.sqrt(10.0)
    b_init = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)

    lc1 = np.pi / dk1
    lc2 = np.pi / dk2

    period1 = 2 * lc1
    width_sfg = 3 * lc2
    period2 = 2 * width_sfg

    # Calculate maximum possible SHG cycles
    max_shg_cycles = int(TOTAL_LENGTH / period1)
    print(f"Sweeping 0 to {max_shg_cycles} SHG cycles...")

    # JAX compatible structure generation
    @jax.jit
    def compute_amp_for_cycles(n_shg_cycles):
        # 1. Generate Structure
        actual_shg_len = n_shg_cycles * period1

        # Calculate domain counts
        n_shg_domains = n_shg_cycles * 2

        rem_len = TOTAL_LENGTH - actual_shg_len
        n_sfg_cycles = jnp.maximum(0, jnp.floor(rem_len / period2).astype(int))
        n_sfg_domains = n_sfg_cycles * 2

        idx = jnp.arange(MAX_N)

        # Determine masks
        is_shg = idx < n_shg_domains
        is_sfg = (idx >= n_shg_domains) & (idx < (n_shg_domains + n_sfg_domains))

        # Assign widths
        w = jnp.zeros(MAX_N, dtype=jnp.float64)
        w = jnp.where(is_shg, lc1, w)
        w = jnp.where(is_sfg, width_sfg, w)

        # 2. Simulate
        s = jnp.tile(jnp.array([1.0, -1.0]), (MAX_N // 2 + 1))[:MAX_N]
        k_shg = s * k_val_shg
        k_sfg = s * k_val_sfg

        # Prevent simulation if we exceeded max_N (basic check, though we padded enough)
        # In this vectorization, MAX_N=25000 is plenty.

        b_res = cwes2.simulate_twm(w, k_shg, k_sfg, dk1, dk2, b_init)
        return jnp.abs(b_res[2]), w

    # Vectorize over integer cycles
    cycles_arr = jnp.arange(max_shg_cycles + 1)

    t0 = time.time()
    amps, widths = jax.vmap(compute_amp_for_cycles)(cycles_arr)

    # Wait for execution
    amps.block_until_ready()
    print(f"Simulation took: {time.time() - t0:.2f}s")

    best_idx = jnp.argmax(amps)
    best_amp = float(amps[best_idx])
    best_w = widths[best_idx]
    best_cycles = int(cycles_arr[best_idx])

    print(f"Best Amp: {best_amp:.6f}")
    print(f"Best Params: SHG Cycles {best_cycles}")

    # Remove padding zeros for saving
    w_clean = best_w[best_w > 0]

    with open("best_ratio.pkl", "wb") as f:
        pickle.dump({"widths": np.array(w_clean), "amp": best_amp, "shg_cycles": best_cycles}, f)


if __name__ == "__main__":
    main()
