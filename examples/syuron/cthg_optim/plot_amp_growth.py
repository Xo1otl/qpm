import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Enable x64 for precision (matches other scripts)
jax.config.update("jax_enable_x64", val=True)

from qpm import cwes2, mgoslt

# Ensure 64-bit precision if needed, matching other scripts
jax.config.update("jax_enable_x64", val=False)


@dataclass
class SimulationConfig:
    shg_len: float = 10000
    sfg_len: float = 7500
    kappa_shg: float = 1.5e-5 / (2 / jnp.pi)
    kappa_sfg: float = 1.5e-5 / (2 / jnp.pi) * 2
    temperature: float = 70.0
    wavelength: float = 1.064
    input_power: float = 10.0


def get_structure_params(cfg: SimulationConfig, widths: jax.Array) -> tuple[jax.Array, jax.Array, float, float]:
    """
    Reconstructs kappa arrays and delta_k values based on the widths and config.
    """
    # Calculate wave vectors and mismatch
    dk_shg = mgoslt.calc_twm_delta_k(cfg.wavelength, cfg.wavelength, cfg.temperature)
    dk_sfg = mgoslt.calc_twm_delta_k(cfg.wavelength, cfg.wavelength / 2, cfg.temperature)

    # Reconstruct signs.
    # maximize_amp.py constructs initial structure with alternating signs: +1, -1, +1, -1...
    # We assume the optimization preserves the number of domains, just changes widths.
    num_domains = len(widths)
    # Signs flip: +1, -1, +1, -1...
    # We can generate this pattern easily.
    signs = jnp.array([1.0 if i % 2 == 0 else -1.0 for i in range(num_domains)])

    # Kappa arrays
    k_shg_vals = signs * cfg.kappa_shg
    k_sfg_vals = signs * cfg.kappa_sfg

    return k_shg_vals, k_sfg_vals, dk_shg, dk_sfg


def main():
    parser = argparse.ArgumentParser(description="Plot amplitude growth from optimized structure.")
    parser.add_argument("filename", type=str, nargs="?", default="amp_opt_result.pkl", help="Path to the file containing widths (.pkl or .npy).")
    parser.add_argument("--save", type=str, default="amp_growth.png", help="Output filename for the plot.")
    parser.add_argument("--save-structure", type=str, default=None, help="Output filename for the structure plot (e.g., structure.png).")
    args = parser.parse_args()

    print(f"Loading from: {args.filename}")
    path = Path(args.filename)
    if not path.exists():
        print(f"Error: File {args.filename} not found.")
        return

    cfg = SimulationConfig()

    if path.suffix == ".pkl":
        with path.open("rb") as f:
            data = pickle.load(f)

        # Check if it is the dict structure we expect
        if isinstance(data, dict) and "params" in data:
            widths = data["params"]
            # Try to recover config if present
            if "config" in data:
                loaded_cfg = data["config"]
                # We can't directly assign the object because the class definition might be slightly different
                # (e.g. methods vs no methods, or different module scope).
                # But here it is likely the same or compatible data class.
                # Let's map fields if possible, or just print what we found.
                # Ideally, we should use the loaded config values.

                # Careful: The SimulationConfig in the pkl might be from maximize_amp_lbfgs.py
                # which has fields like 'iterations' that are not in this script's SimulationConfig.
                # We should extract the common physical params.
                cfg.shg_len = getattr(loaded_cfg, "shg_len", cfg.shg_len)
                cfg.sfg_len = getattr(loaded_cfg, "sfg_len", cfg.sfg_len)
                cfg.kappa_shg = getattr(loaded_cfg, "kappa_shg", cfg.kappa_shg)
                cfg.kappa_sfg = getattr(loaded_cfg, "kappa_sfg", cfg.kappa_sfg)
                cfg.temperature = getattr(loaded_cfg, "temperature", cfg.temperature)
                cfg.wavelength = getattr(loaded_cfg, "wavelength", cfg.wavelength)
                cfg.input_power = getattr(loaded_cfg, "input_power", cfg.input_power)

            print("Loaded data from pickle.")
        else:
            # Fallback if it's just widths in pkl? Unlikely per my change.
            print("Unknown pickle format, expecting dict with 'params'.")
            return

    else:
        # Assume npy
        widths = jnp.load(args.filename)
        print("Loaded widths from npy.")

    # Ensure widths are absolute
    widths = jnp.abs(widths)

    print("Configuration:")
    print(cfg)

    print(f"Structure: {len(widths)} domains")
    print(f"Total Length: {jnp.sum(widths):.2f} um")

    # Reconstruct params
    k_shg, k_sfg, dk1, dk2 = get_structure_params(cfg, widths)

    # Initial State
    amp_fund = jnp.sqrt(cfg.input_power)
    b_initial = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)

    # Run Simulation with Trace
    print("Running simulation with trace...")
    start_time = time.perf_counter()
    _, trace = cwes2.simulate_twm_with_trace(widths, k_shg, k_sfg, dk1, dk2, b_initial)
    trace.block_until_ready()
    print(f"Simulation took: {time.perf_counter() - start_time:.4f} s")

    # Process Trace
    trace_np = np.array(trace)
    a1 = trace_np[:, 0]
    a2 = trace_np[:, 1]
    a3 = trace_np[:, 2]

    # Position array
    z_full = np.concatenate([jnp.array([0.0]), jnp.cumsum(widths)])
    z_full_np = np.array(z_full)

    # Plotting
    print(f"Plotting to {args.save}...")
    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(10, 6))

    plt.plot(z_full_np, np.abs(a1), label=r"$1\omega$", color="blue", linewidth=2)
    plt.plot(z_full_np, np.abs(a2), label=r"$2\omega$", color="orange", linewidth=2)
    plt.plot(z_full_np, np.abs(a3), label=r"$3\omega$", color="green", linewidth=2)

    plt.xlabel("Position (um)")
    plt.ylabel("Amplitude (a.u.)")
    plt.title("Amplitude Growth")
    plt.grid(True, linestyle=":")
    plt.legend()

    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(args.save)
    print(f"Plot saved to {args.save}")

    if args.save_structure:
        print(f"Plotting structure to {args.save_structure}...")
        plt.figure(figsize=(10, 6))
        # widths is in microns
        plt.plot(np.arange(len(widths)), widths, marker="o", markersize=2, linestyle="-", linewidth=0.5)
        plt.xlabel("Domain Index")
        plt.ylabel("Domain Width (µm)")
        plt.title("Domain Structure")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.savefig(args.save_structure)
        print(f"Structure plot saved to {args.save_structure}")

    print("Done.")


if __name__ == "__main__":
    main()
