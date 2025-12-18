from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata

from qpm import wgmode

# Constants
C_LIGHT_UMS = 2.99792458e14  # Speed of light in um/s
EPS0 = 8.854187e-18  # Vacuum permittivity in F/um


@dataclass
class KappaConfig:
    """Configuration for Kappa calculation."""

    fund_wavelength: float
    shg_wavelength: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int
    d33_val: float


def interpolate_field(result: wgmode.ModeResult, grid_depth: np.ndarray, grid_width: np.ndarray) -> np.ndarray:
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
        # print(f"    -> Mode dominant component: {dominant_idx} (Energy ratios: {energies / np.max(energies)})")
        values_mesh = v_reshaped[dominant_idx, :]
    elif values_mesh.size != points_mesh.shape[0]:
        # print(f"    Warning: Basis size mismatch ({values_mesh.size} vs {points_mesh.shape[0]}). Truncating.")
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
    config: KappaConfig,
) -> tuple[float, float, complex, np.ndarray]:
    """Calculates the overlap integral and kappa coefficient."""

    # Nonlinearity Profile (D33 only in the core, x >= 0)
    d_profile = np.where(xx >= 0, config.d33_val, 0.0)

    omega_2 = 2 * np.pi * C_LIGHT_UMS / config.shg_wavelength

    # Note: Ensure conjugate is on the generated field (SHG)
    integrand = np.conj(e_shg) * d_profile * (e_fund**2)

    overlap_integral = np.sum(integrand) * area_elem
    kappa_complex = (omega_2 * EPS0 / 4.0) * overlap_integral

    return overlap_integral, np.abs(kappa_complex), kappa_complex, integrand
