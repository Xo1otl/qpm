from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from solve_with_femwell import ModeResult, SimulationConfig, compute_modes_from_config, find_tm00_mode

# Constants
C_LIGHT_UMS = 2.99792458e14  # Speed of light in um/s
EPS0 = 8.854187e-18  # Vacuum permittivity in F/um
D33_VAL = 1.38e-5  # Nonlinear coefficient in um/V
PLOT_FILENAME = "kappa_calculation_debug.png"


@dataclass
class KappaConfig:
    fund_wavelength: float = 1.031
    shg_wavelength: float = 0.5155
    x_min: float = -2.0
    x_max: float = 12.0
    y_min: float = -15.0
    y_max: float = 15.0
    nx: int = 300
    ny: int = 400


def simulate_wavelength(wavelength: float) -> ModeResult | None:
    print(f"\nSimulating for wavelength {wavelength} um...")
    cfg = SimulationConfig(wavelength_um=wavelength, plot_modes=False)
    _, modes = compute_modes_from_config(cfg)
    return find_tm00_mode(modes)


def interpolate_field(result: ModeResult, grid_depth: np.ndarray, grid_width: np.ndarray) -> np.ndarray:
    """
    Extracts field data from FEM basis and interpolates onto a regular grid.

    Args:
        result: The mode result containing field data.
        grid_depth: Meshgrid array for depth (x).
        grid_width: Meshgrid array for width (y).

    Returns:
        Interpolated electric field array.
    """
    basis = result.field_data.basis
    # basis.doflocs is [width, depth], so we transpose to (N, 2)
    # The columns are [width (y), depth (x)]
    points_mesh = basis.doflocs.T
    values_mesh = result.field_data.E

    # Handle Vector Modes (blocked data [Ex, Ey, Ez])
    if values_mesh.size == 3 * points_mesh.shape[0]:
        v_reshaped = values_mesh.reshape(3, -1)
        energies = np.sum(np.abs(v_reshaped) ** 2, axis=1)
        dominant_idx = np.argmax(energies)
        print(f"    -> Mode dominant component: {dominant_idx} (Energy ratios: {energies / np.max(energies)})")
        values_mesh = v_reshaped[dominant_idx, :]
    elif values_mesh.size != points_mesh.shape[0]:
        print(f"    Warning: Basis size mismatch ({values_mesh.size} vs {points_mesh.shape[0]}). Truncating.")
        values_mesh = values_mesh[: points_mesh.shape[0]]

    # width corresponds to y, depth corresponds to x
    # points_mesh is [y, x]
    # We query at (grid_width, grid_depth)
    return griddata(points_mesh, values_mesh, (grid_width, grid_depth), method="linear", fill_value=0.0)


def compute_overlap(
    e_fund: np.ndarray,
    e_shg: np.ndarray,
    xx: np.ndarray,
    area_elem: float,
    shg_wavelength: float,
) -> tuple[float, float, complex, np.ndarray]:
    """Calculates the overlap integral and kappa coefficient."""

    # Nonlinearity Profile (D33 only in the core, x >= 0)
    d_profile = np.where(xx >= 0, D33_VAL, 0.0)

    omega_2 = 2 * np.pi * C_LIGHT_UMS / shg_wavelength

    # Note: Ensure conjugate is on the generated field (SHG)
    integrand = np.conj(e_shg) * d_profile * (e_fund**2)

    overlap_integral = np.sum(integrand) * area_elem
    kappa_complex = (omega_2 * EPS0 / 4.0) * overlap_integral

    return overlap_integral, np.abs(kappa_complex), kappa_complex, integrand


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
    cfg = KappaConfig()

    # 1. Simulate
    tm00_fund = simulate_wavelength(cfg.fund_wavelength)
    if not tm00_fund:
        print("Error: Could not find TM00 mode for fundamental wavelength.")
        return

    tm00_shg = simulate_wavelength(cfg.shg_wavelength)
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
    e_fund = interpolate_field(tm00_fund, xx, yy)
    print("    Interpolating SHG field...")
    e_shg = interpolate_field(tm00_shg, xx, yy)

    # 4. Compute
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    area_elem = dx * dy

    overlap, kappa_abs, kappa_c, integrand = compute_overlap(e_fund, e_shg, xx, area_elem, cfg.shg_wavelength)

    print("\n--- Result ---")
    print(f"Overlap Integral: {overlap:.4e}")
    print(f"Kappa_SHG: {kappa_abs:.6e}")
    print(f"Kappa_SHG (Complex): {kappa_c:.4e}")

    # 5. Plot
    plot_kappa_vis(x_grid, y_grid, e_fund, e_shg, integrand)


if __name__ == "__main__":
    run_kappa_calculation()
