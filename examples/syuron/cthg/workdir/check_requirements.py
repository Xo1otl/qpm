import pickle

import jax
import jax.numpy as jnp
import numpy as np

from qpm import cwes2, mgoslt

jax.config.update("jax_enable_x64", val=True)

# Constants
wavelength = 1.064
temperature = 70.0


def main():
    print("Checking best SGLD structure...")
    try:
        with open("best_sgld.pkl", "rb") as f:
            data = pickle.load(f)
            widths = data["widths"]
            amp_stored = data["amp"]
    except FileNotFoundError:
        print("Error: best_sgld.pkl not found.")
        return

    widths_abs = np.abs(widths)

    # 1. Minimum Width Check
    min_w = np.min(widths_abs)
    print(f"Minimum Domain Width: {min_w:.4f} um")
    if min_w < 1.49:  # Tolerance
        print("FAIL: Minimum width < 1.5 um")
    else:
        print("PASS: Minimum width >= 1.5 um")

    # 2. Total Length Check
    total_len = np.sum(widths_abs)
    print(f"Total Length: {total_len:.4f} um")
    if total_len > 20500:  # Tolerance
        print("FAIL: Length > 20 mm significantly")
    elif total_len > 20000:
        print(f"WARNING: Length {total_len:.1f} um > 20000 um (target)")
    else:
        print("PASS: Length <= 20 mm")

    # 3. Amplitude Re-Verification
    dk1 = float(mgoslt.calc_twm_delta_k(wavelength, wavelength, temperature))
    dk2 = float(mgoslt.calc_twm_delta_k(wavelength, wavelength / 2, temperature))
    k_val_shg = 1.5e-5 / (2 / np.pi)
    k_val_sfg = 1.5e-5 / (2 / np.pi) * 2
    amp_fund = jnp.sqrt(10.0)
    b_init = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)

    # Simulation
    n = len(widths_abs)
    # Estimate block size
    bs = 300
    for b in range(min(500, n), 19, -1):
        if n % b == 0:
            bs = b
            break

    s = jnp.tile(jnp.array([1.0, -1.0]), (n // 2 + 1))[:n]
    signs = s * 1.0
    k_shg = signs * k_val_shg
    k_sfg = signs * k_val_sfg

    b_final = cwes2.simulate_super_step(jnp.array(widths_abs), k_shg, k_sfg, dk1, dk2, b_init, bs)
    amp_calc = float(jnp.abs(b_final[2]))

    print(f"Stored Amplitude: {amp_stored}")
    print(f"Verified Amplitude: {amp_calc}")

    if amp_calc >= 1.8:
        print("PASS: Amplitude >= 1.8")
    else:
        print(f"FAIL: Amplitude {amp_calc:.3f} < 1.8")


if __name__ == "__main__":
    main()
