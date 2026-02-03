from dataclasses import dataclass
from typing import TYPE_CHECKING
import time

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
    resolutions_nm: tuple[float, ...] = (0, 10, 25, 50, 100)  # 0 means ideal


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


def quantize_widths(widths: jnp.ndarray, dx_um: float) -> jnp.ndarray:
    """Quantizes the domain widths by snapping boundaries to a grid."""
    if dx_um <= 0:
        return widths

    # Calculate cumulative positions (boundaries)
    boundaries = jnp.cumsum(widths)

    # Snap boundaries to grid
    quantized_boundaries = jnp.round(boundaries / dx_um) * dx_um

    # Recalculate widths
    # Prepend 0 to calculate diff
    padded_boundaries = jnp.concatenate((jnp.array([0.0]), quantized_boundaries))
    new_widths = jnp.diff(padded_boundaries)

    # Ensure no negative widths
    new_widths = jnp.maximum(new_widths, 0.0)

    return new_widths


def quantize_dithered_duty(d_n: jnp.ndarray, Lp: float, dx_um: float) -> jnp.ndarray:
    """
    Quantizes duty cycles using error diffusion on the nonlinearity sin(pi*d).
    This aims to preserve the effective nonlinearity magnitude on average.
    """
    # Available discrete widths (assuming max width = Lp)
    # We allow width > Lp? No, max duty 1.0.
    num_steps = int(jnp.ceil(Lp / dx_um))
    possible_widths = jnp.arange(num_steps + 1) * dx_um
    # Cap at Lp
    possible_widths = possible_widths[possible_widths <= Lp + 1e-9]

    possible_duties = possible_widths / Lp
    possible_effs = jnp.sin(jnp.pi * possible_duties)

    target_effs = jnp.sin(jnp.pi * d_n)

    # Error Diffusion Loop (Serial, unfortunately, but num length 10000 is fast in CPU)
    # Use JAX scan? Or simple numpy/python list is fine for 10000.
    # scan is better.

    def scan_body(carry, target):
        accum_error = carry

        # We want to match (target + error)
        desired = target + accum_error

        # Find closest available efficiency
        # Using abs diff
        diffs = jnp.abs(possible_effs - desired)
        path_idx = jnp.argmin(diffs)

        chosen_eff = possible_effs[path_idx]
        chosen_duty = possible_duties[path_idx]

        # New error
        new_error = desired - chosen_eff

        return new_error, chosen_duty

    _, d_n_dithered = jax.lax.scan(scan_body, 0.0, target_effs)

    return d_n_dithered


def run_simulation(
    widths: jnp.ndarray,
    kappas: jnp.ndarray,
    dks: jnp.ndarray,
) -> jnp.ndarray:
    """Runs the SHG simulation."""
    b_initial = jnp.array(1.0 + 0.0j)
    batch_simulate = jax.jit(jax.vmap(cwes.simulate_shg_npda, in_axes=(None, None, 0, None)))
    # For large arrays (10000 periods -> 30000 widths), JIT might be slow or hit memory limits.
    # Use block_until_ready() if benchmarking?
    return batch_simulate(widths, kappas, dks, b_initial)


def calculate_metrics(target: jnp.ndarray, actual: jnp.ndarray) -> tuple[float, float, float]:
    """Calculates Absolute RMSE, Normalized RMSE, and Peak Ratio."""
    target_peak = jnp.max(jnp.abs(target))
    actual_peak = jnp.max(jnp.abs(actual))

    # Absolute RMSE
    abs_rmse = float(jnp.sqrt(jnp.mean((jnp.abs(target) - jnp.abs(actual)) ** 2)))

    # Normalized RMSE (Shape distortion)
    target_norm = target / target_peak
    actual_norm = actual / actual_peak
    norm_rmse = float(jnp.sqrt(jnp.mean((jnp.abs(target_norm) - jnp.abs(actual_norm)) ** 2)))

    return abs_rmse, norm_rmse, float(actual_peak)


def plot_results(
    wls: jnp.ndarray,
    results_dict: dict[str, jnp.ndarray],
    resolutions: list[float],
    norm_rmses: list[float],
    efficiency_ratios: list[float],
) -> None:
    """Plots the normalized simulation results and RMSE trend."""

    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.1,
        subplot_titles=("Normalized Amplitude (Shape Comparison)", "Efficiency (Peak Amplitude)", "Shape RMSE vs Quantization Step"),
    )

    # Reference Key
    ref_key = "Ideal" if "Ideal" in results_dict else list(results_dict.keys())[0]
    ref_peak = jnp.max(jnp.abs(results_dict[ref_key]))

    # Plot Normalized Spectra (Row 1)
    # Target (Ideal)
    fig.add_trace(
        go.Scatter(
            x=wls,
            y=jnp.abs(results_dict[ref_key]) / ref_peak,
            mode="lines",
            name="Ideal (Norm)",
            line={"color": "black", "width": 3, "dash": "solid"},
        ),
        row=1,
        col=1,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    color_idx = 0

    for label, amps in results_dict.items():
        if label == "Ideal":
            continue

        peak = jnp.max(jnp.abs(amps))

        color = colors[color_idx % len(colors)]
        color_idx += 1

        fig.add_trace(
            go.Scatter(
                x=wls,
                y=jnp.abs(amps) / peak,
                mode="lines",
                name=f"{label} (Norm)",
                line={"color": color, "width": 1.5, "dash": "dot"},
            ),
            row=1,
            col=1,
        )

    # Plot Efficiency (Row 2) - Peak Ampl vs Resolution
    # Filter out ideal (0 res) for x-axis plot, or include it at 0

    fig.add_trace(
        go.Scatter(
            x=resolutions,
            y=efficiency_ratios,
            mode="lines+markers",
            name="Peak Amplitude",
            marker={"size": 10, "color": "blue"},
            line={"color": "blue"},
        ),
        row=2,
        col=1,
    )

    # Plot Normalized RMSE (Row 3)
    res_plot = [r for r in resolutions if r > 0]
    rmse_plot = [rmse for r, rmse in zip(resolutions, norm_rmses) if r > 0]

    fig.add_trace(
        go.Scatter(
            x=res_plot,
            y=rmse_plot,
            mode="lines+markers",
            name="Shape RMSE",
            marker={"size": 10, "color": "red"},
            line={"color": "red"},
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=1200,
        width=1000,
        title="Normalized Quantization Effects (Shape & Efficiency)",
    )

    fig.update_xaxes(title_text="Wavelength (µm)", row=1, col=1)
    fig.update_yaxes(title_text="Norm. Amplitude", row=1, col=1)

    fig.update_xaxes(title_text="Quantization Step (nm)", row=2, col=1)
    fig.update_yaxes(title_text="Peak Amplitude", row=2, col=1)

    fig.update_xaxes(title_text="Quantization Step (nm)", row=3, col=1)
    fig.update_yaxes(title_text="RMSE (Normalized)", row=3, col=1)

    fig.write_html("quantization_effects.html")
    print("Results exported to quantization_effects.html")


def main() -> None:
    config = SimulationConfig()
    print(f"Resolutions to test: {config.resolutions_nm} nm")
    print(f"Num Periods: {config.num_periods}")

    # 1. Physics & Geometry Setup
    dk_val = mgoslt.calc_twm_delta_k(config.design_wl, config.design_wl, config.design_temp)
    Lp = 2 * (jnp.pi / dk_val)
    Lc = Lp / 2
    print(f"Lp (Coherence Length * 2): {Lp:.4f} um")
    print(f"Lc (Coherence Length): {Lc:.4f} um")

    # Dynamic resolution update
    Lc_nm = Lc * 1000.0
    config.resolutions_nm = (0.0, 100.0, Lc_nm / 4, Lc_nm / 2, Lc_nm)
    print(f"Testing resolutions (nm): {[f'{r:.2f}' for r in config.resolutions_nm]}")

    # 2. Generate Profile
    d_n, sign_profile = generate_gaussian_profile(config.num_periods, Lp, config.spatial_sigma_ratio)

    # 3. Simulation parameters
    wls = jnp.linspace(config.wl_start, config.wl_end, config.wl_points)
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)

    # 4. Construct Base Geometry AND Run Simulations

    results = {}
    norm_rmses = []
    efficiency_vals = []
    resolutions_plot = []

    # Ideal
    print("Simulating Ideal...")
    widths_ideal, kappas = construct_geometry(d_n, sign_profile, Lp, config.kappa_mag)
    amps_ideal = run_simulation(widths_ideal, kappas, dks)
    amps_ideal.block_until_ready()

    results["Ideal"] = amps_ideal
    _, n_rmse, peak = calculate_metrics(amps_ideal, amps_ideal)
    norm_rmses.append(n_rmse)
    efficiency_vals.append(peak)
    resolutions_plot.append(0.0)
    print(f"  Ideal Peak: {peak:.4e}")

    for res_nm in config.resolutions_nm:
        if res_nm == 0:
            continue

        dx_um = res_nm / 1000.0

        # 1. Standard Quantization (Snap widths)
        label_std = f"Std {res_nm:.0f}nm"
        print(f"Simulating {label_std}...")
        w_quant = quantize_widths(widths_ideal, dx_um)
        amps_std = run_simulation(w_quant, kappas, dks)
        amps_std.block_until_ready()

        results[label_std] = amps_std
        _, n_rmse_std, peak_std = calculate_metrics(amps_ideal, amps_std)
        # Store for plot? We need a way to group them.
        # For now, let's just plot them all.

        print(f"  [Std] Norm RMSE: {n_rmse_std:.4e} | Peak: {peak_std:.4e}")

        # Export if this is the most quantized one (Lc)
        if res_nm == config.resolutions_nm[-1]:
            export_path = "domain_structure_Lc_std.npy"
            jnp.save(export_path, {"widths": w_quant, "kappas": kappas})
            print(f"Exported Standard domain structure to {export_path}")

        # 2. Dithered Quantization
        label_dith = f"Dith {res_nm:.0f}nm"
        print(f"Simulating {label_dith}...")

        # Dither the DUTY CYCLES
        d_n_dith = quantize_dithered_duty(d_n, Lp, dx_um)

        # Recalculate geometries with dithered duties
        w_dith_raw, kappas_dith = construct_geometry(d_n_dith, sign_profile, Lp, config.kappa_mag)

        # Export if this is the most quantized one (Lc)
        if res_nm == config.resolutions_nm[-1]:
            export_path = "domain_structure_Lc_dith.npy"
            jnp.save(export_path, {"widths": w_dith_raw, "kappas": kappas_dith})
            print(f"Exported Dithered domain structure to {export_path}")

        amps_dith = run_simulation(w_dith_raw, kappas_dith, dks)
        amps_dith.block_until_ready()

        results[label_dith] = amps_dith
        _, n_rmse_dith, peak_dith = calculate_metrics(amps_ideal, amps_dith)

        print(f"  [Dith] Norm RMSE: {n_rmse_dith:.4e} | Peak: {peak_dith:.4e}")

        # Add to metrics arrays for dot plots (maybe just Std for now in summary line?)
        # Let's add both to plot manually or modify plot_results to filter?
        # We will update plot_results to handle mixed keys.

        # Hack for the line-plot arrays: append Std values to keep line continuous
        norm_rmses.append(n_rmse_std)
        efficiency_vals.append(peak_std)
        resolutions_plot.append(res_nm)

    # 5. Plot (Updated to handle colors better?)
    plot_results(wls, results, resolutions_plot, norm_rmses, efficiency_vals)

    # 6. Plot Domain Structure (Zoom) for the Coarsest Resolution (Lc)
    # This answers "How did you do it?"

    # Find the largest resolution
    res_nm = config.resolutions_nm[-1]
    dx_um = res_nm / 1000.0
    print(f"\nGenering domain structure plot for Res: {res_nm:.0f} nm...")

    # 1. Ideal
    d_ideal = d_n

    # 2. Standard
    w_std = quantize_widths(widths_ideal, dx_um)
    # Extract pulse widths (indices 1, 4, 7...)
    # widths_ideal was constructed as [gap/2, pulse, gap/2] for each period.
    # So len is 3 * num_periods.
    d_std = w_std[1::3] / Lp

    # 3. Dithered
    d_dith = quantize_dithered_duty(d_n, Lp, dx_um)

    plot_domain_structure(d_ideal, d_std, d_dith, res_nm)


def plot_domain_structure(
    d_ideal: jnp.ndarray,
    d_std: jnp.ndarray,
    d_dith: jnp.ndarray,
    res_nm: float,
) -> None:
    """Plots a zoomed-in view of the domain structure."""

    # Zoom region: slope of the Gaussian.
    # Center is N/2. Sigma is N/8.
    # Steepest slope is around N/2 +/- Sigma.
    N = len(d_ideal)
    center = N // 2
    sigma = N // 8

    zoom_start = center - sigma - 100
    zoom_end = center - sigma + 100
    idx = jnp.arange(zoom_start, zoom_end)

    # Nonlinearity (Efficiency) = sin(pi * D)
    eff_ideal = jnp.sin(jnp.pi * d_ideal)
    eff_std = jnp.sin(jnp.pi * d_std)
    eff_dith = jnp.sin(jnp.pi * d_dith)

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(f"Duty Cycle Profile (Zoom) @ {res_nm:.0f} nm", "Effective Nonlinearity (sin(πD))"),
        vertical_spacing=0.1,
    )

    # Row 1: Duty Cycle
    fig.add_trace(go.Scatter(x=idx, y=d_ideal[idx], name="Ideal", line=dict(color="black", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=d_std[idx], name="Standard (Round)", line=dict(color="red", width=1.5, shape="hv")), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=d_dith[idx], name="Dithered", line=dict(color="green", width=1.5, shape="hv")), row=1, col=1)

    # Row 2: Efficiency
    fig.add_trace(go.Scatter(x=idx, y=eff_ideal[idx], name="Ideal Eff", line=dict(color="black", width=3), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=idx, y=eff_std[idx], name="Std Eff", line=dict(color="red", width=1.5, shape="hv"), showlegend=False), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=idx, y=eff_dith[idx], name="Dith Eff", line=dict(color="green", width=1.5, shape="hv"), showlegend=False), row=2, col=1
    )

    fig.update_layout(height=800, width=1000, title=f"Domain Structure Visualization (Resolution: {res_nm:.0f} nm)")
    fig.update_xaxes(title_text="Period Index", row=2, col=1)
    fig.update_yaxes(title_text="Duty Cycle", row=1, col=1)
    fig.update_yaxes(title_text="sin(πD)", row=2, col=1)

    fig.write_html("quantization_structure.html")
    print("Structure plot exported to quantization_structure.html")


if __name__ == "__main__":
    main()
