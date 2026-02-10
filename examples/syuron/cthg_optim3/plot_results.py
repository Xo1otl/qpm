import argparse
import pickle

import japanize_matplotlib  # pyright: ignore[reportUnusedImport]
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from qpm import cwes2, mgoslt

# Enable x64 (matches user's change in optimization script)
jax.config.update("jax_enable_x64", True)


def setup_font_scaling(scalar):
    """Scales all matplotlib font sizes by a scalar."""
    base_size = 12
    scaled_size = base_size * scalar

    plt.rc("font", size=scaled_size)
    plt.rc("axes", titlesize=scaled_size * 1.2)
    plt.rc("axes", labelsize=scaled_size)
    plt.rc("xtick", labelsize=scaled_size)
    plt.rc("ytick", labelsize=scaled_size)
    plt.rc("legend", fontsize=scaled_size)
    plt.rc("figure", titlesize=scaled_size * 1.5)


def reconstruct_sim_params(mask_len):
    """Reconstructs simulation parameters (k1, k2, dk1, dk2, b_init)"""
    wl = 1.064
    T = 70.0
    dk1 = mgoslt.calc_twm_delta_k(wl, wl, T)
    dk2 = mgoslt.calc_twm_delta_k(wl, wl / 2, T)

    kappa_shg_val = 1.5e-5 / (2 / jnp.pi)
    b_init = jnp.array([jnp.sqrt(10.0), 0.0, 0.0], dtype=jnp.complex128)

    signs = np.tile([1.0, -1.0], mask_len // 2 + 1)[:mask_len]

    k1 = jnp.array(signs * kappa_shg_val)
    k2 = jnp.array(signs * 2 * kappa_shg_val)

    return k1, k2, dk1, dk2, b_init


def run_trace_simulation(w_clean, block_size=10):
    """Runs the simulation with trace and returns trace (amps) and Z coordinates."""
    n = len(w_clean)
    k1, k2, dk1, dk2, b_init = reconstruct_sim_params(n)

    _, trace = cwes2.simulate_magnus_with_trace(jnp.array(w_clean), k1, k2, dk1, dk2, b_init, block_size=block_size)

    trace_np = np.array(trace)
    amps = np.abs(trace_np)

    cumulative_z = np.cumsum(w_clean)

    n_steps = trace_np.shape[0]
    z_indices = np.arange(1, n_steps + 1) * block_size
    z_indices = np.minimum(z_indices, n)
    z_trace = cumulative_z[z_indices - 1]

    # Prepend 0
    z_trace = np.insert(z_trace, 0, 0.0)

    # Initial amplitude
    amp0 = np.abs(np.array(b_init))
    amps_full = np.vstack([amp0, amps])

    return z_trace, amps_full


def construct_noro_structure():
    """Constructs the Noro structure (2.5mm SHG, 7.5mm SFG)."""
    wl = 1.064
    T = 70.0
    dk1 = mgoslt.calc_twm_delta_k(wl, wl, T)
    dk2 = mgoslt.calc_twm_delta_k(wl, wl / 2, T)

    w_shg = float(jnp.pi / dk1)
    w_sfg = float(jnp.pi / dk2)

    sfg_len_um = 7500.0
    shg_len_um = 10000.0

    n_shg = round(shg_len_um / w_shg)
    n_sfg = round(sfg_len_um / w_sfg)

    w_list = [w_shg] * n_shg + [w_sfg] * n_sfg
    return np.array(w_list)


def plot_structure(ax, w, title):
    """Plots domain width vs z-coordinate (um) using POINTS."""
    # Compute cumulative sum of widths to get z positions in microns
    z_um = np.cumsum(w)

    # Using scatter/plot with 'o' marker
    ax.plot(z_um, w, "o", markersize=0.5, linestyle="None")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 6)
    ax.set_title(title)
    ax.set_ylabel(r"ドメイン幅 ($\mu$m)")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))


def plot_amplitudes(ax, z, amps, title):
    """Plots FW, SHW, THW vs Z."""
    ax.plot(z, amps[:, 0], color="red", label="基本波")
    ax.plot(z, amps[:, 1], color="green", label="SH波")
    ax.plot(z, amps[:, 2], color="purple", label="TH波")

    ax.set_xlabel(r"z軸 ($\mu$m)")
    ax.set_ylabel("振幅発展")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot optimization results.")
    parser.add_argument("--file", type=str, required=True, help="Path to pkl file.")
    parser.add_argument("--scalar", type=float, default=1.0, help="Font size scaling factor.")
    parser.add_argument("--export-csv", action="store_true", help="Export plotted data to CSV files.")
    args = parser.parse_args()

    setup_font_scaling(args.scalar)

    print(f"Loading '{args.file}'...")
    with open(args.file, "rb") as f:
        data = pickle.load(f)

    w_init = data["initial_structure"]
    w_final = data["final_structure"]

    print("Simulating Initial Structure...")
    z_init, amps_init = run_trace_simulation(w_init, block_size=100)

    print("Simulating Final Structure...")
    z_final, amps_final = run_trace_simulation(w_final, block_size=100)

    # --- Plotting - Separate Sets ---

    # Set 1: Initial
    fig_init, axs_init = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True, sharex=True)
    plot_structure(axs_init[0], w_init, "最適化前の構造")
    plot_amplitudes(axs_init[1], z_init, amps_init, "振幅発展")

    fig_init_name = "analysis_initial.png"
    fig_init.savefig(fig_init_name, dpi=300)
    print(f"Saved {fig_init_name}")

    # Set 2: Final
    fig_final, axs_final = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True, sharex=True)
    plot_structure(axs_final[0], w_final, "最適化された構造")
    plot_amplitudes(axs_final[1], z_final, amps_final, "振幅発展")

    fig_final_name = "analysis_final.png"
    fig_final.savefig(fig_final_name, dpi=300)
    print(f"Saved {fig_final_name}")

    # Set 3: Noro (2.5mm SHG + 7.5mm SFG)
    w_noro = construct_noro_structure()
    print("Simulating Noro Structure...")
    z_noro, amps_noro = run_trace_simulation(w_noro, block_size=1)

    fig_noro, axs_noro = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True, sharex=True)
    plot_structure(axs_noro[0], w_noro, "先行研究のタンデム構造")
    plot_amplitudes(axs_noro[1], z_noro, amps_noro, "振幅発展")
    fig_noro.savefig("fig_noro.png", dpi=300)

    print(f"Final Max THG:   {np.max(amps_final[:, 2]):.4f}")
    print("Noro THG = 1.704!")

    if args.export_csv:
        print("\nExporting data to CSV...")

        # Helper to save
        def save_structure(name, w):
            z_um = np.cumsum(w)
            data = np.column_stack((z_um, w))
            np.savetxt(f"{name}_structure.csv", data, delimiter=",", header="z_um,width_um", comments="")
            print(f"Saved {name}_structure.csv")

        def save_evolution(name, z, amps):
            data = np.column_stack((z, amps))
            np.savetxt(f"{name}_evolution.csv", data, delimiter=",", header="z_um,fw_amp,sh_amp,th_amp", comments="")
            print(f"Saved {name}_evolution.csv")

        # Initial
        save_structure("initial", w_init / 1000)
        save_evolution("initial", z_init / 1000, amps_init)

        # Final
        save_structure("final", w_final / 1000)
        save_evolution("final", z_final / 1000, amps_final)

        # Noro
        save_structure("noro", w_noro / 1000)
        save_evolution("noro", z_noro / 1000, amps_noro)


if __name__ == "__main__":
    main()
