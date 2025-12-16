from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from jax.scipy.special import erf

from qpm import mgoslt


# --- Data Structures ---
@dataclass
class ProcessParams:
    """Parameters for the waveguide fabrication process."""

    temp_c: float
    d_pe_coeff: float  # Diffusivity for Proton Exchange (um^2/h)
    t_pe_hours: float  # Time for Proton Exchange (h)
    mask_width_um: float  # Width of the mask opening (um)
    t_anneal_hours: float  # Annealing time (h)
    d_x_coeff: float  # Annealing Diffusivity depth (um^2/h)
    d_y_coeff: float  # Annealing Diffusivity width (um^2/h)


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


# --- Domain Logic ---
def new_standard_process_params() -> ProcessParams:
    """Initializes the standard process parameters based on experimental data."""
    return ProcessParams(
        temp_c=70.0,
        d_pe_coeff=0.045,
        t_pe_hours=8.0,
        mask_width_um=50.0,
        t_anneal_hours=100.0,
        d_x_coeff=1.3,
        d_y_coeff=1.3 / 1.5,
    )


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
    - Boundary Condition: Reflecting (Neumann) at x=0.
    - Solution: Product of 1D error function solutions (Cx * Cy).
    """
    d_pe = calculate_initial_depth(params)
    width = params.mask_width_um
    t_diff = params.t_anneal_hours

    # Verticla Diffusion (x direction - depth)
    lx = 2.0 * jnp.sqrt(params.d_x_coeff * t_diff)
    # Boundary at x=0 is reflecting, so we simulate a source from -d_PE to d_PE
    # C_x = 0.5 * [erf((d_PE - x)/Lx) + erf((d_PE + x)/Lx)]
    # Note: The original code had (d_PE + x) which corresponds to the image source at -d_PE to 0
    # ensuring dC/dx = 0 at x=0.
    c_x = 0.5 * (erf((d_pe - x) / lx) + erf((d_pe + x) / lx))

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


# --- Visualization ---
def create_index_heatmap(result: RefractiveIndexResult, x_coords: jax.Array, y_coords: jax.Array) -> go.Figure:
    """Generates a heatmap of the index profile."""
    # Convert JAX arrays to Numpy for Plotly
    z_np = np.array(result.n_profile)
    x_ax = np.array(y_coords)  # Width is typically x-axis in plots
    y_ax = np.array(x_coords)  # Depth is typically y-axis in plots

    fig = go.Figure(data=go.Heatmap(z=z_np, x=x_ax, y=y_ax, colorscale="Viridis", colorbar={"title": "Refractive Index"}, reversescale=False))

    fig.update_layout(
        title=f"Refractive Index Distribution @ {result.wl_um} um, {result.temp_c}°C (n_sub={result.n_sub:.4f})",
        xaxis_title="Width (µm)",
        yaxis_title="Depth (µm)",
        yaxis={"autorange": "reversed"},  # Depth increases downwards
        width=800,
        height=600,
    )
    return fig


# --- Main Execution ---
def main() -> None:
    print("--- QPM Index Construction (Refactored) ---")

    params = new_standard_process_params()

    # Derived parameters for reporting
    d_pe = calculate_initial_depth(params)

    print("Parameters:")
    print(f"  d_PE (calculated): {d_pe:.4f} um")
    print(f"  W: {params.mask_width_um} um")
    print(f"  t_anneal: {params.t_anneal_hours} h")
    print(f"  D_x: {params.d_x_coeff} um^2/h")
    print(f"  D_y: {params.d_y_coeff:.4f} um^2/h")
    print(f"  Temp: {params.temp_c} C")

    # Wavelengths
    wl_fund = 1.031
    wl_sh = wl_fund / 2.0

    # Calculate Substrate Indices
    n_sub_fund = mgoslt.sellmeier_n_eff(wl_fund, params.temp_c)
    n_sub_sh = mgoslt.sellmeier_n_eff(wl_sh, params.temp_c)

    print("\nSubstrate Indices:")
    print(f"  n_sub(@{wl_fund}um): {n_sub_fund:.6f}")
    print(f"  n_sub(@{wl_sh:.4f}um): {n_sub_sh:.6f}")

    # Check Peak Index at (0,0)
    # We use scalar coordinates for spot checks
    peak_fund_res = calculate_index_profile(SimulationGrid(x_depth=jnp.array(0.0), y_width=jnp.array(0.0)), wl_fund, params)
    peak_sh_res = calculate_index_profile(SimulationGrid(x_depth=jnp.array(0.0), y_width=jnp.array(0.0)), wl_sh, params)

    # We need to extract the scalar value from the 0-d array
    val_fund_0 = peak_fund_res.n_profile
    val_sh_0 = peak_sh_res.n_profile

    print("\nPeak Index (x=0, y=0) after Annealing:")
    print(f"  n(@{wl_fund}um): {val_fund_0:.6f} (Delta: {val_fund_0 - n_sub_fund:.6f})")
    print(f"  n(@{wl_sh:.4f}um): {val_sh_0:.6f} (Delta: {val_sh_0 - n_sub_sh:.6f})")

    # Sample Grid Point
    x_sample = 5.0
    y_sample = 0.0
    sample_res = calculate_index_profile(SimulationGrid(x_depth=jnp.array(x_sample), y_width=jnp.array(y_sample)), wl_fund, params)
    print(f"\nIndex at x={x_sample}, y={y_sample}:")
    print(f"  n(@{wl_fund}um): {sample_res.n_profile:.6f}")

    # Generate Plot
    # Define grid
    x_vec = jnp.linspace(0, 40, 200)
    y_vec = jnp.linspace(-40, 40, 200)
    y_grid, x_grid = jnp.meshgrid(y_vec, x_vec)

    grid = SimulationGrid(x_depth=x_grid, y_width=y_grid)

    result_fund = calculate_index_profile(grid, wl_fund, params)

    plot_filename = "index_distribution_fund.html"
    print(f"\nGenerating plot for Fundamental wave to {plot_filename}...")

    fig = create_index_heatmap(result_fund, x_vec, y_vec)
    fig.show()


if __name__ == "__main__":
    main()
