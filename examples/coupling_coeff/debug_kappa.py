import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np
from qpm import config, cwes, wgmode


def run_debug_calculation():
    print("--- Debug Kappa Question ---")
    process_params = config.new_process_params()
    cfg = config.new_kappa_config()

    print("1. Computing Fundamental Mode...")
    # Only compute fundamental mode to save time if we are testing E_fund with E_fund
    tm00_fund = wgmode.compute_tm00(config.new_simulation_config(cfg.fund_wavelength, process_params))
    if not tm00_fund:
        print("Error: Could not find TM00 mode.")
        return

    print("2. Grid Setup & Interpolation...")
    x_grid = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y_grid = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    yy, xx = np.meshgrid(y_grid, x_grid, indexing="xy")

    e_fund = cwes.interpolate_field(tm00_fund, xx, yy)

    # Grid area
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    area_elem = dx * dy

    print("3. Computing Overlap (E_fund vs E_fund)...")
    # Passing e_fund as both arguments
    overlap, kappa_abs, kappa_c, integrand = cwes.compute_overlap(e_fund, e_fund, xx, area_elem, cfg)

    print("\n--- Comparative Results ---")
    print(f"d33 Value (Material Property): {cfg.d33_val:.6e} [units in code]")
    print(f"Calculated Overlap Integral:   {overlap:.6e}")
    print(f"Calculated Kappa (abs):        {kappa_abs:.6e}")

    # Check ratio
    if overlap != 0:
        print(f"Ratio (Overlap / d33):         {overlap / cfg.d33_val:.4f}")
    if cfg.d33_val != 0:
        print(f"Ratio (d33 / Overlap):         {cfg.d33_val / overlap:.4f}")


if __name__ == "__main__":
    run_debug_calculation()
