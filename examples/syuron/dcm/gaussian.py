from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from qpm import cwes, mgoslt

if TYPE_CHECKING:
    pass

jax.config.update("jax_enable_x64", val=True)


@dataclass
class SimulationConfig:
    """Configuration for the SHG simulation."""

    num_periods: int = 10000
    design_wl: float = 1.064
    design_temp: float = 70.0
    wl_start: float = 1.0638
    wl_end: float = 1.0642
    wl_points: int = 1000
    kappa_mag: float = 1.31e-5 / (2 / jnp.pi)
    spatial_sigma_ratio: float = 8.0  # L_total / this_ratio


def generate_gaussian_profile(num_periods: int, Lp: float, sigma_ratio: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generates the target Gaussian profile values and polarity signs."""
    L_total = num_periods * Lp

    # Spatial grid centered at 0
    z_period_centers = (jnp.arange(num_periods) + 0.5) * Lp
    z_n = z_period_centers - L_total / 2.0

    # Gaussian spatial width
    spatial_sigma = L_total / sigma_ratio

    # Target: Gaussian(z)
    target_profile = jnp.exp(-(z_n**2) / (2 * (spatial_sigma**2)))

    # Normalize to max 1.0
    norm_profile = target_profile / jnp.max(jnp.abs(target_profile))

    # Magnitudes (D) and Signs
    # Inverse Nonlinearity Map: D = arcsin(M) / pi
    d_n = jnp.arcsin(jnp.abs(norm_profile)) / jnp.pi
    sign_profile = jnp.sign(norm_profile)

    return d_n, sign_profile


def construct_geometry(d_n: jnp.ndarray, sign_profile: jnp.ndarray, Lp: float, kappa_mag: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Constructs the 3-domain (center-anchored) geometry."""
    num_periods = len(d_n)

    # Structure: [Gap/2, Pulse, Gap/2]
    # Center-anchored pulse
    gap_widths = (1 - d_n) * Lp / 2.0
    pulse_widths = d_n * Lp

    # Interleave [Gap, Pulse, Gap]
    widths = jnp.column_stack((gap_widths, pulse_widths, gap_widths)).ravel()

    # Polarity pattern: (+, -, +) for [Gap(+), Pulse(-), Gap(+)]
    base_signs = jnp.tile(jnp.array([1.0, -1.0, 1.0]), num_periods)
    period_signs = jnp.repeat(sign_profile, 3)
    kappas = kappa_mag * base_signs * period_signs

    return widths, kappas


def run_simulation(
    widths: jnp.ndarray,
    kappas: jnp.ndarray,
    dks: jnp.ndarray,
) -> jnp.ndarray:
    """Runs the SHG simulation."""
    b_initial = jnp.array(1.0 + 0.0j)
    batch_simulate = jax.jit(jax.vmap(cwes.simulate_shg_npda, in_axes=(None, None, 0, None)))
    print("Running simulation...")
    return batch_simulate(widths, kappas, dks, b_initial)


def calculate_theoretical_spec_shape(config: SimulationConfig, wls: jnp.ndarray, spatial_sigma: float) -> jnp.ndarray:
    """Calculates the theoretical Gaussian spectral shape (normalized)."""
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)
    dk_center = mgoslt.calc_twm_delta_k(config.design_wl, config.design_wl, config.design_temp)

    # Gaussian Spectrum E ~ exp(-(dk * sigma)^2 / 2)
    shift_dks = dks - dk_center
    shape = jnp.exp(-((shift_dks * spatial_sigma) ** 2) / 2)
    return shape


def calculate_metrics(
    wls: jnp.ndarray,
    target_shape: jnp.ndarray,
    amps: jnp.ndarray,
) -> tuple[float, float]:
    """Calculates RMSE and Peak GDD."""
    # RMSE
    vals_norm = jnp.abs(amps) / jnp.max(jnp.abs(amps))
    rmse = float(jnp.sqrt(jnp.mean((vals_norm - target_shape) ** 2)))

    # GDD Calculation
    # Convert to Angular Frequency (rad/ps)
    c_um_ps = 299.792458
    omegas = 2 * jnp.pi * c_um_ps / wls

    # Sort for differentiation
    idx_sorted = jnp.argsort(omegas)
    omegas_sorted = omegas[idx_sorted]
    amps_sorted = amps[idx_sorted]

    # Unwrap Phase -> GD -> GDD
    phi = jnp.unwrap(jnp.angle(amps_sorted))

    d_omega = jnp.diff(omegas_sorted)
    gd = jnp.diff(phi) / d_omega
    omega_mid = (omegas_sorted[1:] + omegas_sorted[:-1]) / 2.0

    d_gd = jnp.diff(gd)
    d_omega_mid = jnp.diff(omega_mid)
    gdd = d_gd / d_omega_mid

    # Take GDD at center
    peak_gdd = float(gdd[len(gdd) // 2])

    return rmse, peak_gdd


def plot_results(
    wls: jnp.ndarray,
    amps: jnp.ndarray,
    target: jnp.ndarray,
    d_n: jnp.ndarray,
    Lp: float,
    metrics: tuple[float, float],
) -> None:
    """Plots the simulation results and metrics."""
    rmse, peak_gdd = metrics

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        subplot_titles=("Amplitude", "Phase", "Duty Ratio Profile"),
    )

    # 1. Amplitude (Spectrum)
    # Target
    fig.add_trace(
        go.Scatter(
            x=wls,
            y=target,
            mode="lines",
            name="Target",
            line={"color": "green", "width": 2, "dash": "dash"},
        ),
        row=1,
        col=1,
    )
    # Simulation
    fig.add_trace(
        go.Scatter(
            x=wls,
            y=jnp.abs(amps),
            mode="lines",
            name="Simulated",
            line={"color": "blue", "width": 2},
        ),
        row=1,
        col=1,
    )

    # 2. Phase
    fig.add_trace(
        go.Scatter(
            x=wls,
            y=jnp.unwrap(jnp.angle(amps)),
            mode="lines",
            name="Phase",
            line={"color": "blue", "width": 1},
        ),
        row=2,
        col=1,
    )

    # 3. Duty Ratio
    z_axis = (jnp.arange(len(d_n)) + 0.5) * Lp
    fig.add_trace(
        go.Scatter(
            x=z_axis,
            y=d_n,
            mode="lines",
            name="Duty Ratio",
            line={"color": "#d62728", "width": 1.5},
        ),
        row=3,
        col=1,
    )

    metrics_str = f"RMSE={rmse:.4e} | GDD={peak_gdd:.4e} ps²"
    fig.update_layout(
        height=1000,
        width=900,
        title={
            "text": f"Inverse Design: Gaussian Spectrum<br>{metrics_str}",
            "x": 0.5,
            "xanchor": "center",
        },
        margin={"t": 100},
    )
    fig.update_xaxes(title_text="Wavelength (µm)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Phase (rad)", row=2, col=1)
    fig.update_xaxes(title_text="Position (µm)", row=3, col=1)
    fig.update_yaxes(title_text="Duty Ratio", row=3, col=1)

    fig.write_html("gaussian_results.html")
    print("Results exported to gaussian_results.html")


def main() -> None:
    config = SimulationConfig()

    # 1. Physics & Geometry Setup
    dk_val = mgoslt.calc_twm_delta_k(config.design_wl, config.design_wl, config.design_temp)
    Lp = 2 * (jnp.pi / dk_val)
    L_total = config.num_periods * Lp
    spatial_sigma = L_total / config.spatial_sigma_ratio

    # 2. Generate Profile
    d_n, sign_profile = generate_gaussian_profile(config.num_periods, Lp, config.spatial_sigma_ratio)

    # 3. Simulation parameters
    wls = jnp.linspace(config.wl_start, config.wl_end, config.wl_points)
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)

    # 4. Construct & Run
    widths, kappas = construct_geometry(d_n, sign_profile, Lp, config.kappa_mag)
    amps = run_simulation(widths, kappas, dks)

    # 5. Metrics & Target
    # Compute theoretical shape and scale it to simulation peak for visual comparison
    target_shape = calculate_theoretical_spec_shape(config, wls, spatial_sigma)
    rmse, peak_gdd = calculate_metrics(wls, target_shape, amps)

    # Scale target for plotting
    target_plot = target_shape * jnp.max(jnp.abs(amps))

    print(f"Metrics: RMSE={rmse:.4e}, Peak GDD={peak_gdd:.4e} ps²")

    # 6. Plot
    plot_results(wls, amps, target_plot, d_n, Lp, (rmse, peak_gdd))


if __name__ == "__main__":
    main()
