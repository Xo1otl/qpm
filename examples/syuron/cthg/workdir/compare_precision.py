import pickle
import numpy as np
import jax
import jax.numpy as jnp
from qpm import cwes2, mgoslt

jax.config.update("jax_enable_x64", val=True)

wavelength = 1.064
temperature = 70.0


def check(filename):
    print(f"Checking {filename}...")
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            w = data["widths"]
            amp_stored = data["amp"]
    except:
        print("Fail load.")
        return

    n = len(w)

    dk1 = float(mgoslt.calc_twm_delta_k(wavelength, wavelength, temperature))
    dk2 = float(mgoslt.calc_twm_delta_k(wavelength, wavelength / 2, temperature))
    k_val_shg = 1.5e-5 / (2 / np.pi)
    k_val_sfg = 1.5e-5 / (2 / np.pi) * 2
    amp_fund = jnp.sqrt(10.0)
    b_init = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)

    s = jnp.tile(jnp.array([1.0, -1.0]), (n // 2 + 1))[:n]
    signs = s * 1.0
    k_shg = signs * k_val_shg
    k_sfg = signs * k_val_sfg

    # Precise
    b_final = cwes2.simulate_twm(jnp.array(w), k_shg, k_sfg, dk1, dk2, b_init)
    amp_precise = float(jnp.abs(b_final[2]))

    print(f"Stored: {amp_stored:.6f}")
    print(f"Precise: {amp_precise:.6f}")
    print(f"Delta: {amp_precise - amp_stored:.6f}")


def main():
    check("optimized_2nd_order.pkl")
    check("best_polished_long.pkl")


if __name__ == "__main__":
    main()
