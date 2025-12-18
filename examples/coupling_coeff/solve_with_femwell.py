import os

# Force JAX to use CPU to avoid GPU contention (since FEM is CPU-bound anyway)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import matplotlib.pyplot as plt
import numpy as np

from qpm import ape, wgmode


def verify_single_mode_condition(results: wgmode.ModeList) -> None:
    guided_modes = [m for m in results if m.is_guided]
    guided_count = len(guided_modes)

    print(f"Verification: Found {guided_count} guided modes.")

    if guided_count == 0:
        print("FAIL: No modes guided.")
    elif guided_count == 1:
        print("SUCCESS: Single-Mode operation achieved (Fundamental only).")
    else:
        print(f"FAIL: Multi-mode behavior detected ({guided_count} modes guided).")
        for m in guided_modes:
            print(f"  - Mode {m.index}: n_eff={m.n_eff:.6f}")


def plot_mode_result(ctx: wgmode.SimulationContext, result: wgmode.ModeResult) -> None:
    if not result.is_guided:
        print(f"Mode {result.index}: n_eff = {result.n_eff:.6f} (Cutoff - Ignored)")
        return

    print(f"Mode {result.index}: n_eff = {result.n_eff:.6f} (Guided)")

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot A: Refractive Index ---
    n_core_max = np.max(ctx.n_dist)
    n_core_min = np.min(ctx.n_dist)
    print(f"Refractive Index Range: min={n_core_min:.5f}, max={n_core_max:.5f}")

    ctx.basis.plot(
        ctx.n_dist,
        ax=axes[0],
        cmap="viridis",
        shading="gouraud",
        # Use data limits to maximize contrast
        vmin=n_core_min,
        vmax=n_core_max,
    )

    mappable = axes[0].collections[-1]
    plt.colorbar(mappable, ax=axes[0], label="n")
    axes[0].set_title(f"Refractive Index (Clamped to {ctx.n_sub:.3f}+)")
    axes[0].set_aspect("equal")

    # --- Plot B: Mode Intensity ---
    result.field_data.plot_intensity(ax=axes[1], colorbar=True)
    axes[1].set_title(f"Mode {result.index} Intensity (n_eff={result.n_eff:.5f})")
    axes[1].set_aspect("equal")
    axes[1].set_ylabel("")

    plt.tight_layout()
    filename = f"out/sim_mode_{result.index}.png"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()


def run() -> None:
    print("--- RUNNING WAVEGUIDE SIMULATION ---")

    # 1. Initialization
    pp = ape.new_default_process_params()
    pp.is_buried = True
    cfg = wgmode.SimulationConfig()
    cfg.process_params = pp

    # Expand domain for buried waveguide to avoid truncation
    # calculate_index.py uses +/- 50, so we use similar range
    cfg.depth_min = -50.0
    cfg.depth_max = 50.0
    cfg.width_min = -50.0
    cfg.width_max = 50.0

    # 2. Computation
    ctx, modes = wgmode.compute_modes_from_config(cfg)
    print(f"Substrate Index (n_sub): {ctx.n_sub:.6f}")

    # 3. Visualization
    if cfg.plot_modes:
        guided_modes = [m for m in modes if m.is_guided]
        if guided_modes:
            print(f"Plotting {len(guided_modes)} guided modes...")
            for mode in guided_modes:
                plot_mode_result(ctx, mode)
        else:
            print("Warning: No guided modes found to plot.")

    # 4. Verification
    verify_single_mode_condition(modes)


if __name__ == "__main__":
    run()
