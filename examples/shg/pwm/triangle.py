from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from plotly import graph_objects as go

from qpm import cwes, mgoslt

if TYPE_CHECKING:
    from collections.abc import Callable

# Enable x64 for precision
jax.config.update("jax_enable_x64", val=True)


@dataclass
class SimulationConfig:
    """Configuration for the SHG simulation."""

    num_periods: int = 10000
    design_wl: float = 1.031
    design_temp: float = 70.0
    wl_start: float = 1.025
    wl_end: float = 1.037
    wl_points: int = 1000
    kappa_mag: float = 1.31e-5 / (2 / jnp.pi)
    spatial_scale_factor: float = 20.0


def generate_triangle_profile(num_periods: int, Lp: float, scale_factor: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generates the target Sinc^2 profile values (for Triangle spectrum) and polarity signs."""
    L_total = num_periods * Lp

    # Spatial grid centered at 0
    z_period_centers = (jnp.arange(num_periods) + 0.5) * Lp
    z_center = L_total / 2.0
    z_n = z_period_centers - z_center

    # Target: Sinc^2 (Fourier Pair of Triangle)
    # sinc_arg scaling derived from original script: z_n * scale / (L/2)
    sinc_arg = z_n * scale_factor / (L_total / 2.0)
    target_profile = jnp.sinc(sinc_arg) ** 2

    # Normalize to max 1.0
    max_val = jnp.max(jnp.abs(target_profile))
    norm_profile = target_profile / max_val

    # Magnitudes (D) and Signs
    mag_profile = jnp.abs(norm_profile)
    # For sinc^2, sign is effectively always +1 (as sinc^2 >= 0)
    sign_profile = jnp.sign(norm_profile)

    # Inverse Nonlinearity Map: D = arcsin(M) / pi
    d_n = jnp.arcsin(mag_profile) / jnp.pi

    return d_n, sign_profile


def construct_geometry_2d(d_n: jnp.ndarray, sign_profile: jnp.ndarray, Lp: float, kappa_mag: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Constructs the 2-domain (left-anchored) geometry."""
    num_periods = len(d_n)

    # Structure: [Pulse, Gap] (Left-Anchored)
    # Interleave [Pulse, Gap]
    widths_2d = jnp.column_stack((d_n * Lp, (1 - d_n) * Lp)).ravel()

    # Polarity pattern: (+, -) * sign_profile
    base_signs = jnp.tile(jnp.array([1.0, -1.0]), num_periods)
    period_signs = jnp.repeat(sign_profile, 2)
    kappas_2d = kappa_mag * base_signs * period_signs

    return widths_2d, kappas_2d


def construct_geometry_3d(d_n: jnp.ndarray, sign_profile: jnp.ndarray, Lp: float, kappa_mag: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Constructs the 3-domain (center-anchored) geometry."""
    num_periods = len(d_n)

    # Structure: [Gap/2, Pulse, Gap/2]
    gap_widths = (1 - d_n) * Lp / 2.0
    pulse_widths = d_n * Lp

    # Interleave [Gap, Pulse, Gap]
    widths_3d = jnp.column_stack((gap_widths, pulse_widths, gap_widths)).ravel()

    # Polarity pattern: (-, +, -) * sign_profile -> Inverted to (+, -, +) for [Gap(+), Pulse(-), Gap(+)]
    base_signs = jnp.tile(jnp.array([1.0, -1.0, 1.0]), num_periods)
    period_signs = jnp.repeat(sign_profile, 3)
    kappas_3d = kappa_mag * base_signs * period_signs

    return widths_3d, kappas_3d


def run_simulation(
    widths: jnp.ndarray,
    kappas: jnp.ndarray,
    dks: jnp.ndarray,
) -> jnp.ndarray:
    """Runs the SHG simulation."""
    b_initial = jnp.array(1.0 + 0.0j)

    # JIT compiled simulation function
    batch_simulate = jax.jit(jax.vmap(cwes.simulate_shg_npda, in_axes=(None, None, 0, None)))

    print("Running simulation...")
    amps = batch_simulate(widths, kappas, dks, b_initial)
    print("Running simulation...")
    amps = batch_simulate(widths, kappas, dks, b_initial)
    return jnp.abs(amps)


def calculate_theoretical_spec_shape(config: SimulationConfig, wls: jnp.ndarray, L_total: float) -> jnp.ndarray:
    """Calculates the theoretical Triangle spectral shape (normalized to 1.0)."""
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)
    dk_center = mgoslt.calc_twm_delta_k(config.design_wl, config.design_wl, config.design_temp)
    dk_half_width = 4 * jnp.pi * config.spatial_scale_factor / L_total
    shift_dks = dks - dk_center
    # Triangle function: max(0, 1 - |x/w|)
    return jnp.maximum(0.0, 1.0 - jnp.abs(shift_dks / dk_half_width))


def calculate_rmse(measured_vals: jnp.ndarray, target_shape: jnp.ndarray) -> float:
    """Calculates RMSE between normalized measured values and target shape."""
    vals_norm = measured_vals / jnp.max(measured_vals)
    return float(jnp.sqrt(jnp.mean((vals_norm - target_shape) ** 2)))


def plot_results(
    wls: jnp.ndarray,
    results_map: dict[str, tuple[jnp.ndarray, float]],
    target_spectra: dict[str, jnp.ndarray],
) -> None:
    """Plots the simulation results and metrics."""
    fig = go.Figure()

    # Plot Targets
    colors = ["green", "purple"]
    for i, (name, target) in enumerate(target_spectra.items()):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=wls,
                y=target,
                mode="lines",
                name=name,
                line={"color": color, "width": 2, "dash": "dash"},
            ),
        )

    # Plot Simulations
    for name, (amps, _) in results_map.items():
        color = "blue" if "3-Domain" in name else "red"
        dash = "solid" if "3-Domain" in name else "dot"

        fig.add_trace(
            go.Scatter(
                x=wls,
                y=amps,
                mode="lines",
                name=name,
                line={"color": color, "width": 2, "dash": dash},
            ),
        )

    # Create Metrics Title
    metrics_strs = [f"[{name.split()[0]}: RMSE={rmse:.4e}]" for name, (_, rmse) in results_map.items()]
    title_suffix = "<br>" + "<br>".join(metrics_strs)

    fig.update_layout(
        height=600,
        width=900,
        title_text=f"Inverse Design: Triangular Spectrum (Sinc² Apodization) {title_suffix}",
    )
    fig.update_xaxes(title_text="Wavelength (µm)")
    fig.update_xaxes(title_text="Wavelength (µm)")
    fig.update_yaxes(title_text="Amplitude (Normalized)")

    fig.show()


def main() -> None:
    config = SimulationConfig()

    # 1. Physics & Geometry Setup
    dk_val = mgoslt.calc_twm_delta_k(config.design_wl, config.design_wl, config.design_temp)
    Lp = 2 * (jnp.pi / dk_val)
    L_total = config.num_periods * Lp

    # 2. Generate Profile
    d_n, sign_profile = generate_triangle_profile(config.num_periods, Lp, config.spatial_scale_factor)

    # 3. Pre-calculate Simulation space
    wls = jnp.linspace(config.wl_start, config.wl_end, config.wl_points)
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)

    # 4. Calculate Theoretical Target (Shape)
    target_shape = calculate_theoretical_spec_shape(config, wls, L_total)

    # 5. Run Simulations
    cases: dict[str, Callable[[jax.Array, jax.Array, float, float], tuple[jax.Array, jax.Array]]] = {
        "3-Domain (Center-Anchored)": construct_geometry_3d,
        "2-Domain (Left-Anchored)": construct_geometry_2d,
    }

    results: dict[str, tuple[jnp.ndarray, float]] = {}

    for name, constructor in cases.items():
        print(f"Simulating {name}...")
        w, k = constructor(d_n, sign_profile, Lp, config.kappa_mag)
        amps = run_simulation(w, k, dks)
        rmse = calculate_rmse(amps, target_shape)
        results[name] = (amps, rmse)
        print(f"Accuracy {name}: RMSE={rmse:.4e}")

    # 6. Prepare Plot Data (Scale target to Peak)
    ref_amp_3d, _ = results["3-Domain (Center-Anchored)"]
    ref_amp_2d, _ = results["2-Domain (Left-Anchored)"]

    target_3d = target_shape * jnp.max(ref_amp_3d)
    target_2d = target_shape * jnp.max(ref_amp_2d)

    targets = {
        "Target (3D Peak)": target_3d,
        "Target (2D Peak)": target_2d,
    }

    plot_results(wls, results, targets)


if __name__ == "__main__":
    main()
