from dataclasses import dataclass

import jax
import jax.numpy as jnp
import plotly.graph_objects as go

from qpm import ape, config, mgoslt


@dataclass
class SimulationConfig:
    """Configuration for the simulation steps."""

    params: ape.ProcessParams
    wl_fund: float
    wl_sh: float

    @property
    def n_sub_fund(self) -> float:
        return mgoslt.sellmeier_n_eff(self.wl_fund, self.params.temp_c)

    @property
    def n_sub_sh(self) -> float:
        return mgoslt.sellmeier_n_eff(self.wl_sh, self.params.temp_c)


def create_index_heatmap(result: ape.RefractiveIndexResult, x_coords: jax.Array, y_coords: jax.Array, title_suffix: str = "") -> go.Figure:
    """Generates a heatmap of the index profile."""
    fig = go.Figure(
        data=go.Heatmap(
            z=result.n_profile,
            x=y_coords,  # Width is typically x-axis in plots
            y=x_coords,  # Depth is typically y-axis in plots
            colorscale="Viridis",
            colorbar={"title": "Refractive Index"},
            reversescale=False,
        ),
    )

    fig.update_layout(
        title=f"Refractive Index Distribution @ {result.wl_um} um, {result.temp_c}°C{title_suffix} (n_sub={result.n_sub:.4f})",
        xaxis_title="Width (µm)",
        yaxis_title="Depth (µm)",
        yaxis={"autorange": "reversed", "scaleanchor": "x", "scaleratio": 1},
        width=700,
        height=700,
    )
    return fig


def print_process_parameters(params: ape.ProcessParams) -> None:
    """Prints the process parameters."""
    d_pe = ape.calculate_initial_depth(params)
    print("Parameters:")
    print(f"  d_PE (calculated): {d_pe:.4f} um")
    print(f"  W: {params.mask_width_um} um")
    print(f"  t_anneal: {params.t_anneal_hours} h")
    print(f"  D_x: {params.d_x_coeff} um^2/h")
    print(f"  D_y: {params.d_y_coeff:.4f} um^2/h")
    print(f"  Temp: {params.temp_c} C")


def analyze_peak_indices(cfg: SimulationConfig) -> None:
    """Calculates and prints peak indices at (0,0)."""
    # Check Peak Index at (0,0)
    # We use scalar coordinates for spot checks
    grid_zero = ape.SimulationGrid(x_depth=jnp.array(0.0), y_width=jnp.array(0.0))
    peak_fund_res = ape.calculate_index_profile(grid_zero, cfg.wl_fund, cfg.params)
    peak_sh_res = ape.calculate_index_profile(grid_zero, cfg.wl_sh, cfg.params)

    # We need to extract the scalar value from the 0-d array
    val_fund_0 = peak_fund_res.n_profile
    val_sh_0 = peak_sh_res.n_profile

    n_sub_fund = cfg.n_sub_fund
    n_sub_sh = cfg.n_sub_sh

    suffix = " (Buried)" if cfg.params.is_buried else " (Surface)"

    print(f"\nPeak Index (x=0, y=0) after Annealing{suffix}:")
    print(f"  n(@{cfg.wl_fund}um): {val_fund_0:.6f} (Delta: {val_fund_0 - n_sub_fund:.6f})")

    if not cfg.params.is_buried:
        print(f"  n(@{cfg.wl_sh:.4f}um): {val_sh_0:.6f} (Delta: {val_sh_0 - n_sub_sh:.6f})")


def analyze_sample_point(cfg: SimulationConfig, x: float, y: float) -> None:
    """Analyzes index at a specific sample point."""
    sample_res = ape.calculate_index_profile(ape.SimulationGrid(x_depth=jnp.array(x), y_width=jnp.array(y)), cfg.wl_fund, cfg.params)
    print(f"\nIndex at x={x}, y={y}:")
    print(f"  n(@{cfg.wl_fund}um): {sample_res.n_profile:.6f}")


def generate_plot(cfg: SimulationConfig, output_filename: str) -> None:
    """Generates and saves the index distribution plot."""
    # Generate Plot
    x_vec = jnp.linspace(-50, 50, 200) if cfg.params.is_buried else jnp.linspace(0, 50, 200)

    y_vec = jnp.linspace(-50, 50, 200)
    y_grid, x_grid = jnp.meshgrid(y_vec, x_vec)

    grid = ape.SimulationGrid(x_depth=x_grid, y_width=y_grid)

    result_fund = ape.calculate_index_profile(grid, cfg.wl_fund, cfg.params)

    print(f"\nGenerating plot for Fundamental wave to {output_filename}...")

    suffix = " (Buried)" if cfg.params.is_buried else ""
    fig = create_index_heatmap(result_fund, x_vec, y_vec, title_suffix=suffix)
    fig.write_html(output_filename)


def run_surface_simulation(wl_fund: float, wl_sh: float) -> None:
    """Runs the simulation for surface waveguides."""
    params = config.new_process_params()
    params.is_buried = False

    cfg = SimulationConfig(params=params, wl_fund=wl_fund, wl_sh=wl_sh)

    print_process_parameters(params)

    print("\nSubstrate Indices:")
    print(f"  n_sub(@{wl_fund}um): {cfg.n_sub_fund:.6f}")
    print(f"  n_sub(@{wl_sh:.4f}um): {cfg.n_sub_sh:.6f}")

    analyze_peak_indices(cfg)
    analyze_sample_point(cfg, x=5.0, y=0.0)
    generate_plot(cfg, "out/index_distribution_fund.html")


def run_buried_simulation(wl_fund: float, wl_sh: float) -> None:
    """Runs the simulation for buried waveguides."""
    print("\n--- Buried Calculation ---")
    params = config.new_process_params()
    params.is_buried = True

    cfg = SimulationConfig(params=params, wl_fund=wl_fund, wl_sh=wl_sh)

    analyze_peak_indices(cfg)
    generate_plot(cfg, "out/index_distribution_fund_buried.html")


def main() -> None:
    print("--- QPM Index Construction (Refactored) ---")

    # Wavelengths
    wl_fund = 1.031
    wl_sh = wl_fund / 2.0

    run_surface_simulation(wl_fund, wl_sh)
    run_buried_simulation(wl_fund, wl_sh)


if __name__ == "__main__":
    main()
