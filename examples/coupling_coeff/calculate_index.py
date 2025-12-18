import jax
import jax.numpy as jnp
import plotly.graph_objects as go

from qpm import ape, config, mgoslt


# --- Visualization ---
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


def main() -> None:
    print("--- QPM Index Construction (Refactored) ---")

    params = config.new_process_params()
    params.is_buried = False  # Start with surface for the first plot

    # Derived parameters for reporting
    d_pe = ape.calculate_initial_depth(params)

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
    peak_fund_res = ape.calculate_index_profile(ape.SimulationGrid(x_depth=jnp.array(0.0), y_width=jnp.array(0.0)), wl_fund, params)
    peak_sh_res = ape.calculate_index_profile(ape.SimulationGrid(x_depth=jnp.array(0.0), y_width=jnp.array(0.0)), wl_sh, params)

    # We need to extract the scalar value from the 0-d array
    val_fund_0 = peak_fund_res.n_profile
    val_sh_0 = peak_sh_res.n_profile

    print("\nPeak Index (x=0, y=0) after Annealing (Surface):")
    print(f"  n(@{wl_fund}um): {val_fund_0:.6f} (Delta: {val_fund_0 - n_sub_fund:.6f})")
    print(f"  n(@{wl_sh:.4f}um): {val_sh_0:.6f} (Delta: {val_sh_0 - n_sub_sh:.6f})")

    # Sample Grid Point
    x_sample = 5.0
    y_sample = 0.0
    sample_res = ape.calculate_index_profile(ape.SimulationGrid(x_depth=jnp.array(x_sample), y_width=jnp.array(y_sample)), wl_fund, params)
    print(f"\nIndex at x={x_sample}, y={y_sample}:")
    print(f"  n(@{wl_fund}um): {sample_res.n_profile:.6f}")

    # Generate Plot for Surface
    x_vec = jnp.linspace(0, 50, 200)
    y_vec = jnp.linspace(-50, 50, 200)
    y_grid, x_grid = jnp.meshgrid(y_vec, x_vec)

    grid = ape.SimulationGrid(x_depth=x_grid, y_width=y_grid)

    result_fund = ape.calculate_index_profile(grid, wl_fund, params)

    plot_filename = "out/index_distribution_fund.html"
    print(f"\nGenerating plot for Fundamental wave to {plot_filename}...")

    fig = create_index_heatmap(result_fund, x_vec, y_vec)
    fig.write_html(plot_filename)

    # --- Buried Calculation ---
    print("\n--- Buried Calculation ---")
    params_buried = config.new_process_params()
    params_buried.is_buried = True

    # Check Peak Index at (0,0) for Buried
    peak_fund_res_buried = ape.calculate_index_profile(ape.SimulationGrid(x_depth=jnp.array(0.0), y_width=jnp.array(0.0)), wl_fund, params_buried)
    val_fund_0_buried = peak_fund_res_buried.n_profile
    print("Peak Index (x=0, y=0) after Annealing (Buried):")
    print(f"  n(@{wl_fund}um): {val_fund_0_buried:.6f} (Delta: {val_fund_0_buried - n_sub_fund:.6f})")

    # Generate Plot for Buried
    # For buried, we might want to see x < 0 as well to verify diffusion "upwards"
    x_vec_buried = jnp.linspace(-50, 50, 200)  # Start from negative depth
    y_vec_buried = jnp.linspace(-50, 50, 200)
    y_grid_buried, x_grid_buried = jnp.meshgrid(y_vec_buried, x_vec_buried)
    grid_buried = ape.SimulationGrid(x_depth=x_grid_buried, y_width=y_grid_buried)

    result_fund_buried = ape.calculate_index_profile(grid_buried, wl_fund, params_buried)

    plot_filename_buried = "out/index_distribution_fund_buried.html"
    print(f"Generating plot for Buried Fundamental wave to {plot_filename_buried}...")

    fig_buried = create_index_heatmap(result_fund_buried, x_vec_buried, y_vec, title_suffix=" (Buried)")
    fig_buried.write_html(plot_filename_buried)


if __name__ == "__main__":
    main()
