import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import matplotlib.pyplot as plt
import numpy as np

from qpm import config, cwes, wgmode

PLOT_FILENAME = "out/kappa_calculation_debug.png"


def plot_kappa_vis(x_grid: np.ndarray, y_grid: np.ndarray, e_fund: np.ndarray, e_shg: np.ndarray, integrand: np.ndarray) -> None:
    """Generates and saves a visualization of the fields and overlap integrand."""
    print("    Generating plot...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.pcolormesh(y_grid, x_grid, np.abs(e_fund), shading="auto")
    plt.title("|E_fund (Ex)|")
    plt.xlabel("Width (y)")
    plt.ylabel("Depth (x)")
    plt.gca().invert_yaxis()

    plt.subplot(1, 3, 2)
    plt.pcolormesh(y_grid, x_grid, np.abs(e_shg), shading="auto")
    plt.title("|E_shg (Ex)|")
    plt.gca().invert_yaxis()

    plt.subplot(1, 3, 3)
    plt.pcolormesh(y_grid, x_grid, np.abs(integrand), shading="auto", cmap="magma")
    plt.title("|Integrand|")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(PLOT_FILENAME)
    print(f"Saved visualization to {PLOT_FILENAME}")


def run_kappa_calculation() -> None:
    print("--- Kappa Calculation Script ---")
    process_params = config.new_process_params()
    cfg = config.new_kappa_config()

    # 1. Simulate
    tm00_fund = wgmode.compute_tm00(config.new_simulation_config(cfg.fund_wavelength, process_params))
    if not tm00_fund:
        print("Error: Could not find TM00 mode for fundamental wavelength.")
        return

    tm00_shg = wgmode.compute_tm00(config.new_simulation_config(cfg.shg_wavelength, process_params))
    if not tm00_shg:
        print("Error: Could not find TM00 mode for SHG wavelength.")
        return

    # 2. Grid Setup
    print("\nCalculating Overlap Integral...")
    x_grid = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y_grid = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)

    yy, xx = np.meshgrid(y_grid, x_grid, indexing="xy")

    # 3. Interpolate
    print("    Interpolating Fundamental field...")
    e_fund = cwes.interpolate_field(tm00_fund, xx, yy)
    print("    Interpolating SHG field...")
    e_shg = cwes.interpolate_field(tm00_shg, xx, yy)

    # 4. Compute
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    area_elem = dx * dy

    overlap, kappa_abs, kappa_c, integrand = cwes.compute_overlap(e_fund, e_shg, xx, area_elem, cfg)

    print("\n--- Result ---")
    print(f"Overlap Integral: {overlap:.4e}")
    print(rf"Kappa_SHG: {kappa_abs:.6e} $\text{{W}}^{-1 / 2} \mu\text{{m}}^{-1}$")
    print(f"Kappa_SHG (Complex): {kappa_c:.4e}")

    # 5. Plot
    plot_kappa_vis(x_grid, y_grid, e_fund, e_shg, integrand)


if __name__ == "__main__":
    run_kappa_calculation()
