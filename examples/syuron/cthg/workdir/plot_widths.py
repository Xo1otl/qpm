import argparse
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from qpm import cwes2, mgoslt

jax.config.update("jax_enable_x64", val=True)

# Constants
wavelength = 1.064
temperature = 70.0


def main():
    parser = argparse.ArgumentParser(description="Domain width distribution plotter")
    parser.add_argument("filename", type=str, help="Path to the .pkl file")
    args = parser.parse_args()

    filename = args.filename
    print(f"Loading {filename}...")

    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            widths = data["widths"]
            amp_stored = data["amp"]
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except KeyError as e:
        print(f"Error: Missing expected key in pickle file: {e}")
        return

    widths = np.abs(widths)
    n = len(widths)
    total_len = np.sum(widths)

    print(f"Stored Amplitude: {amp_stored}")
    print(f"Number of domains: {n}")
    print(f"Total Length: {total_len:.4f} um")

    # Precise Re-Verification
    print("Verifying amplitude with cwes2.simulate_twm (High Precision)...")
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

    b_final = cwes2.simulate_twm(jnp.array(widths), k_shg, k_sfg, dk1, dk2, b_init)
    amp_precise = float(jnp.abs(b_final[2]))

    print(f"Precise Amplitude: {amp_precise:.6f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(widths, "o", markersize=0.5, alpha=0.9, color="tab:blue", markeredgewidth=0)

    plt.title(f"Domain Width Distribution\nFile: {filename}\nAmp: {amp_precise:.4f} | Length: {total_len / 1000:.3f} mm")
    plt.xlabel("Domain Index")
    plt.ylabel("Domain Width (um)")

    plt.text(
        0.95,
        0.95,
        f"L = {total_len:.1f} um",
        transform=plt.gca().transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.grid(True, alpha=0.2)
    output_plot = filename.replace(".pkl", "_scatter.png")
    plt.savefig(output_plot, dpi=300)
    print(f"Saved plot to {output_plot}")


if __name__ == "__main__":
    main()
