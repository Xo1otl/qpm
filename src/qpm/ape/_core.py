from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from qpm import mgoslt


@dataclass
class ProcessParams:
    """Parameters for the Annealed Proton Exchange (APE) process."""

    temp_c: float
    d_pe_coeff: float  # Diffusivity for Proton Exchange (um^2/h)
    t_pe_hours: float  # Time for Proton Exchange (h)
    mask_width_um: float  # Width of the mask opening (um)
    t_anneal_hours: float  # Annealing time (h)
    d_x_coeff: float  # Annealing Diffusivity depth (um^2/h)
    d_y_coeff: float  # Annealing Diffusivity width (um^2/h)
    is_buried: bool = False  # Whether the structure is buried (infinite boundary)


@dataclass
class SimulationGrid:
    """Grid for simulation coordinates."""

    x_depth: jax.Array
    y_width: jax.Array


@dataclass
class RefractiveIndexResult:
    """Result of refractive index calculation."""

    n_profile: jax.Array
    n_sub: float
    wl_um: float
    temp_c: float


def calculate_initial_depth(params: ProcessParams) -> jax.Array:
    """Calculates the initial depth d_PE = 2 * sqrt(D_PE * t_PE)."""
    return 2.0 * jnp.sqrt(params.d_pe_coeff * params.t_pe_hours)


def get_delta_n0(wl_um: float) -> jax.Array:
    """
    Returns delta_n0 for a given wavelength.
    Uses nearest neighbor lookup:
    - ~1.03 um -> 0.012
    - ~0.515 um -> 0.017 (Using 532nm value for SHG)
    """
    dist_fund = jnp.abs(wl_um - 1.031)
    # Note: 0.5155 is the SHG wavelength of 1.031
    dist_sh = jnp.abs(wl_um - 0.5155)

    # Return 0.012 if closer to fundamental, else 0.017
    return jnp.where(dist_fund < dist_sh, 0.012, 0.017)


def concentration_distribution(x: jax.Array, y: jax.Array, params: ProcessParams) -> jax.Array:
    """
    Calculates the normalized concentration C(x,y)/C0 after diffusion.

    Physics:
    - Initial profile: Rectangular block of width W (y) and depth d_PE (x).
    - Boundary Condition:
        - If not buried: Reflecting (Neumann) at x=0.
        - If buried: Infinite (bulk) medium.
    - Solution: Product of 1D error function solutions (Cx * Cy).
    """
    d_pe = calculate_initial_depth(params)
    width = params.mask_width_um
    t_diff = params.t_anneal_hours

    # Verticla Diffusion (x direction - depth)
    lx = 2.0 * jnp.sqrt(params.d_x_coeff * t_diff)

    c_x = 0.5 * (erf((d_pe - x) / lx) + erf(x / lx)) if params.is_buried else 0.5 * (erf((d_pe - x) / lx) + erf((d_pe + x) / lx))

    # Horizontal Diffusion (y direction - width)
    ly = 2.0 * jnp.sqrt(params.d_y_coeff * t_diff)
    # Source from -W/2 to W/2
    c_y = 0.5 * (erf((width / 2.0 - y) / ly) + erf((width / 2.0 + y) / ly))

    return c_x * c_y


def calculate_index_profile(grid: SimulationGrid, wl_um: float, params: ProcessParams) -> RefractiveIndexResult:
    """
    Calculates the refractive index n(x,y) at a specific wavelength.
    n(x,y) = n_sub(wl, T) + delta_n0(wl) * (C(x,y)/C0)
    """
    # Substrate index
    n_sub = mgoslt.sellmeier_n_eff(wl_um, params.temp_c)

    # Max index change
    delta_n0 = get_delta_n0(wl_um)

    # Normalized concentration
    # If grid.x_depth and grid.y_width are meshgrids, this works directly.
    c_norm = concentration_distribution(grid.x_depth, grid.y_width, params)

    n_profile = n_sub + delta_n0 * c_norm

    return RefractiveIndexResult(n_profile=n_profile, n_sub=n_sub, wl_um=wl_um, temp_c=params.temp_c)
