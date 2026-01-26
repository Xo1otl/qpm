from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from qpm import cwes, mgoslt

if TYPE_CHECKING:
    from collections.abc import Callable

jax.config.update("jax_enable_x64", val=True)


@dataclass
class SimulationConfig:
    """Configuration for the SHG simulation."""

    num_periods: int = 10000
    design_wl: float = 1.031
    design_temp: float = 70.0
    wl_start: float = 1.0308
    wl_end: float = 1.0312
    wl_points: int = 1000
    kappa_mag: float = 1.31e-5 / (2 / jnp.pi)
    spatial_sigma_ratio: float = 8.0  # L_total / this_ratio


def generate_gaussian_profile(num_periods: int, Lp: float, sigma_ratio: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generates the target Gaussian profile values and polarity signs."""
    L_total = num_periods * Lp

    # Spatial grid centered at 0
    z_period_centers = (jnp.arange(num_periods) + 0.5) * Lp
    z_center = L_total / 2.0
    z_n = z_period_centers - z_center

    # Gaussian spatial width
    spatial_sigma = L_total / sigma_ratio

    # Target: Gaussian(z) = exp(-z^2 / (2 * sigma^2))
    target_profile = jnp.exp(-(z_n**2) / (2 * (spatial_sigma**2)))

    # Normalize to max 1.0
    max_val = jnp.max(jnp.abs(target_profile))
    norm_profile = target_profile / max_val

    # Magnitudes (D) and Signs
    mag_profile = jnp.abs(norm_profile)
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
    return batch_simulate(widths, kappas, dks, b_initial)


def calculate_theoretical_spec_shape(config: SimulationConfig, wls: jnp.ndarray, spatial_sigma: float) -> jnp.ndarray:
    """Calculates the theoretical Gaussian spectral shape (normalized to 1.0)."""
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)
    dk_center = mgoslt.calc_twm_delta_k(config.design_wl, config.design_wl, config.design_temp)

    # Gaussian Spectrum E ~ exp(-(dk * sigma)^2 / 2)
    shift_dks = dks - dk_center
    return jnp.exp(-((shift_dks * spatial_sigma) ** 2) / 2)


def calculate_gdd(wls: jnp.ndarray, amps: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculates the Group Delay Dispersion (GDD) from spectral phase.

    GDD = d^2(phi) / d(omega)^2
    """
    # 1. Convert to Angular Frequency (rad/ps)
    # c = 299.792458 um/ps
    c_um_ps = 299.792458
    omegas = 2 * jnp.pi * c_um_ps / wls

    # Sort by omega (ascending) for numerical differentiation
    idx_sorted = jnp.argsort(omegas)
    omegas_sorted = omegas[idx_sorted]
    amps_sorted = amps[idx_sorted]

    # 2. Unwrap Phase
    phi = jnp.unwrap(jnp.angle(amps_sorted))

    # 3. First Derivative (GD)
    d_omega = jnp.diff(omegas_sorted)
    d_phi = jnp.diff(phi)
    gd_mid = d_phi / d_omega
    omega_mid = (omegas_sorted[1:] + omegas_sorted[:-1]) / 2.0

    # 4. Second Derivative (GDD)
    d_omega_mid = jnp.diff(omega_mid)
    d_gd = jnp.diff(gd_mid)
    gdd_mid = d_gd / d_omega_mid
    omega_mid2 = (omega_mid[1:] + omega_mid[:-1]) / 2.0

    return omega_mid2, gdd_mid


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
    # We define "Peak GDD" as the GDD value at the wavelength of peak amplitude
    # to avoid noise in the wings.
    _, gdd_vals = calculate_gdd(wls, amps)

    # Map back to sorted omegas used in GDD...
    # Actually, simpler to just take the GDD at the center of the spectrum
    # or weighted average. Let's start with center index of the GDD array.
    center_idx = len(gdd_vals) // 2
    peak_gdd = float(gdd_vals[center_idx])

    return rmse, peak_gdd


def plot_results(
    wls: jnp.ndarray,
    results_map: dict[str, tuple[jnp.ndarray, float, float]],
    target_spectra: dict[str, jnp.ndarray],
) -> None:
    """Plots the simulation results and metrics."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Amplitude", "Phase"))

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
            row=1,
            col=1,
        )

    # Plot Simulations
    metrics_strs = []
    for name, (amps, rmse, _) in results_map.items():
        color = "blue" if "3-Domain" in name else "red"
        dash = "solid" if "3-Domain" in name else "dot"

        # Amplitude
        fig.add_trace(
            go.Scatter(
                x=wls,
                y=jnp.abs(amps),
                mode="lines",
                name=f"{name} (Amp)",
                line={"color": color, "width": 2, "dash": dash},
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Phase
        fig.add_trace(
            go.Scatter(
                x=wls,
                y=jnp.unwrap(jnp.angle(amps)),
                mode="lines",
                name=f"{name} (Phase)",
                line={"color": color, "width": 1, "dash": dash},
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # metrics_strs.append(f"[{name.split()[0]}: RMSE={rmse:.4e}, GDD={peak_gdd:.4e} ps²]")
        metrics_strs.append(f"{name.split()[0]}: RMSE={rmse:.4e}")

    title_suffix = "<br>" + "<br>".join(metrics_strs)
    fig.update_layout(
        height=800,
        width=900,
        title={
            "text": f"Inverse Design: Gaussian Spectrum (1.0308 - 1.0312 µm) {title_suffix}",
            "y": 0.96,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        margin={"t": 120},
    )
    fig.update_xaxes(title_text="Wavelength (µm)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude (Normalized)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (rad)", row=2, col=1)

    fig.show()


def main() -> None:
    config = SimulationConfig()

    # 1. Physics & Geometry Setup
    dk_val = mgoslt.calc_twm_delta_k(config.design_wl, config.design_wl, config.design_temp)
    Lp = 2 * (jnp.pi / dk_val)
    L_total = config.num_periods * Lp
    spatial_sigma = L_total / config.spatial_sigma_ratio

    # 2. Generate Profile
    d_n, sign_profile = generate_gaussian_profile(config.num_periods, Lp, config.spatial_sigma_ratio)

    # 3. Pre-calculate Simulation space
    wls = jnp.linspace(config.wl_start, config.wl_end, config.wl_points)
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)

    # 4. Calculate Theoretical Target (Shape)
    target_shape = calculate_theoretical_spec_shape(config, wls, spatial_sigma)

    # 5. Run Simulations
    cases: dict[str, Callable[[jax.Array, jax.Array, float, float], tuple[jax.Array, jax.Array]]] = {
        "3-Domain (Center-Anchored)": construct_geometry_3d,
        "2-Domain (Left-Anchored)": construct_geometry_2d,
    }

    results: dict[str, tuple[jnp.ndarray, float, float]] = {}

    for name, constructor in cases.items():
        print(f"Simulating {name}...")
        w, k = constructor(d_n, sign_profile, Lp, config.kappa_mag)
        amps = run_simulation(w, k, dks)
        rmse, peak_gdd = calculate_metrics(wls, target_shape, amps)
        results[name] = (amps, rmse, peak_gdd)
        print(f"Accuracy {name}: RMSE={rmse:.4e}, Peak GDD={peak_gdd:.4e} ps²")

    # 6. Prepare Plot Data (Scale target to 3D peak)
    ref_amp_3d, _, _ = results["3-Domain (Center-Anchored)"]
    ref_amp_2d, _, _ = results["2-Domain (Left-Anchored)"]

    target_3d = target_shape * jnp.max(jnp.abs(ref_amp_3d))
    target_2d = target_shape * jnp.max(jnp.abs(ref_amp_2d))

    targets = {
        "Target (3D Peak)": target_3d,
        "Target (2D Peak)": target_2d,
    }

    plot_results(wls, results, targets)


if __name__ == "__main__":
    main()
