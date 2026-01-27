from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from qpm import cwes, mgoslt

if TYPE_CHECKING:
    from collections.abc import Callable

jax.config.update("jax_enable_x64", val=True)


@dataclass
class SimulationConfig:
    """Configuration for the SHG simulation."""

    num_periods: int = 10000  # Increased for better resolution
    design_wl: float = 1.031
    design_temp: float = 70.0
    wl_start: float = 1.029
    wl_end: float = 1.033
    wl_points: int = 4000
    kappa_mag: float = 1.31e-4 / (2 / jnp.pi)  # Roughly 10x higher to compensate for sinc decay/energy spread
    # Flat-top specific
    target_bandwidth_nm: float = 3  # Target flat-top width in nm
    smoothing_factor: float = 0.1  # Ratio of transition width (sigma) to bandwidth


def generate_sinc_profile(num_periods: int, Lp: float, target_bandwidth_dk: float, sigma_z: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generates the target Sinc profile values and polarity signs.
    Sinc(k z / 2) corresponds to Rect(k).
    Applies Gaussian window (sigma_z) to smooth the spectrum (erf edges).
    """
    L_total = num_periods * Lp

    # Spatial grid centered at 0
    z_period_centers = (jnp.arange(num_periods) + 0.5) * Lp
    z_center = L_total / 2.0
    z_n = z_period_centers - z_center

    # Sinc argument:
    # math.sinc(x) = sin(pi * x) / (pi * x)
    # Target: sin(K * z / 2) / (K * z / 2)
    # Let K * z / 2 = pi * x  =>  x = K * z / (2 * pi)

    sinc_arg_norm = target_bandwidth_dk * z_n / (2 * jnp.pi)
    target_profile = jnp.sinc(sinc_arg_norm)

    # Apply Gaussian Window (Apodization)
    # window = exp(-z^2 / (2 * sigma_z^2))
    window = jnp.exp(-(z_n**2) / (2 * sigma_z**2))
    target_profile = target_profile * window

    # Normalize to max 1.0 (sinc max is 1 at z=0)
    # Note: In practice, we might want to window this to reduce Gibbs ripples,
    # but for pure analytical PoC we keep it raw.
    max_val = jnp.max(jnp.abs(target_profile))
    norm_profile = target_profile / max_val

    # Magnitudes (D) and Signs
    mag_profile = jnp.abs(norm_profile)
    sign_profile = jnp.sign(norm_profile)

    # Inverse Nonlinearity Map: D = arcsin(M) / pi
    d_n = jnp.arcsin(mag_profile) / jnp.pi

    return d_n, sign_profile


def construct_geometry_shifted(d_n: jnp.ndarray, sign_profile: jnp.ndarray, Lp: float, kappa_mag: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Constructs the geometry using the Shifted Pulse Model.

    Positive Phase (sign > 0): Pulse centered. [Gap/2 (+), Pulse (-), Gap/2 (+)]
    Negative Phase (sign < 0): Pulse shifted.  [Pulse/2 (-), Gap (+), Pulse/2 (-)]

    This maintains the background nonlinearity as +1 (Gap) and pulse as -1.
    """
    # Calculate widths based on phase sign
    # If sign > 0: Gap/2 = (1-D)L/2. Pulse = DL.
    # If sign < 0: Pulse/2 = DL/2.   Gap = (1-D)L.

    # w1 (First segment)
    # sign > 0: (1-d)*L/2  (Gap/2)
    # sign < 0: d*L/2      (Pulse/2)
    # Note: sign_profile is +/- 1.0 (or 0.0). Construct mask.
    is_pos = sign_profile > 0

    w1 = jnp.where(is_pos, (1 - d_n) * Lp / 2.0, d_n * Lp / 2.0)

    # w2 (Middle segment) = L - 2*w1
    w2 = Lp - 2 * w1

    # w3 (Last segment) = w1
    w3 = w1

    # Stack and ravel: [w1, w2, w3] for each period
    widths = jnp.column_stack((w1, w2, w3)).ravel()

    # Polarity pattern
    # sign > 0: [+1, -1, +1]  (Gap, Pulse, Gap)
    # sign < 0: [-1, +1, -1]  (Pulse, Gap, Pulse)
    # This is exactly 'sign_profile' for the first segment?
    # No.
    # If pos: +1 (Gap). sign_profile is +1. Match.
    # If neg: -1 (Pulse). sign_profile is -1. Match.
    # So s1 = sign_profile.

    # s2 needs to always be opposite of s1?
    # Pos: -1 (Pulse). s1=+1. So -s1.
    # Neg: +1 (Gap). s1=-1. So -s1.
    # Yes. s2 = -sign_profile.

    s1 = sign_profile
    s2 = -sign_profile
    s3 = sign_profile

    signs = jnp.column_stack((s1, s2, s3)).ravel()
    kappas = kappa_mag * signs

    return widths, kappas


def run_simulation(widths: jnp.ndarray, kappas: jnp.ndarray, dks: jnp.ndarray) -> jnp.ndarray:
    """Runs the SHG simulation."""
    b_initial = jnp.array(1.0 + 0.0j)
    batch_simulate = jax.jit(jax.vmap(cwes.simulate_shg_npda, in_axes=(None, None, 0, None)))
    return batch_simulate(widths, kappas, dks, b_initial)


def calculate_theoretical_spec_shape(config: SimulationConfig, wls: jnp.ndarray, target_bandwidth_dk: float) -> jnp.ndarray:
    """Calculates the theoretical Rect spectral shape (normalized to 1.0) using Error Functions (smoothed)."""
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)
    dk_center = mgoslt.calc_twm_delta_k(config.design_wl, config.design_wl, config.design_temp)

    # Rect width W, Gaussian sigma s.
    # Shape ~ erf((k + W/2)/s) - erf((k - W/2)/s)

    shift_dks = dks - dk_center

    # We need the spectral sigma.
    sigma = config.smoothing_factor * target_bandwidth_dk

    # Arg for Erf
    # 0.5 * (erf((x + W/2) / (sqrt(2)*sigma)) - erf((x - W/2) / (sqrt(2)*sigma)))

    # Ideally should normalize max to 1.0.
    # Analytical max is erf(W / (2*sqrt(2)*sigma)) at center x=0.

    w_half = target_bandwidth_dk / 2.0
    denom = jnp.sqrt(2) * sigma

    term1 = erf((shift_dks + w_half) / denom)
    term2 = erf((shift_dks - w_half) / denom)

    shape = 0.5 * (term1 - term2)

    # Normalize
    max_theoretical = erf(w_half / denom)  # assuming term1 -> erf(W/(sqrt2 s)), term2 -> erf(-W/...) = -erf
    # Actually term1 at 0 is erf(w_half/denom), term2 is erf(-w_half/denom) = -erf(w_half/denom).
    # So sum is 2 * erf(...) -> * 0.5 -> erf(...).

    return shape / max_theoretical


def calculate_metrics(target_shape: jnp.ndarray, amps: jnp.ndarray) -> float:
    """Calculates RMSE."""
    vals_norm = jnp.abs(amps) / (jnp.max(jnp.abs(amps)) + 1e-12)
    rmse = float(jnp.sqrt(jnp.mean((vals_norm - target_shape) ** 2)))
    return rmse


def plot_results(wls: jnp.ndarray, results_map: dict[str, tuple[jnp.ndarray, float]], target_spectra: dict[str, jnp.ndarray]) -> None:
    """Plots the simulation results and metrics."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Amplitude", "Phase"))

    # Plot Targets
    colors = ["green", "purple"]
    for i, (name, target) in enumerate(target_spectra.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(x=wls, y=target, mode="lines", name=name, line={"color": color, "width": 2, "dash": "dash"}), row=1, col=1)

    # Plot Simulations
    metrics_strs = []
    for name, (amps, rmse) in results_map.items():
        color = "blue" if "3-Domain" in name else "red"
        dash = "solid" if "3-Domain" in name else "dot"

        # Amplitude
        fig.add_trace(
            go.Scatter(x=wls, y=jnp.abs(amps), mode="lines", name=f"{name} (Amp)", line={"color": color, "width": 2, "dash": dash}, showlegend=True),
            row=1,
            col=1,
        )

        # Phase (Unwrapped)
        phi = jnp.unwrap(jnp.angle(amps))
        # Remove linear slope (optional, but helps visualization if there's a delay)
        # For now, raw phase
        fig.add_trace(
            go.Scatter(x=wls, y=phi, mode="lines", name=f"{name} (Phase)", line={"color": color, "width": 1, "dash": dash}, showlegend=False),
            row=2,
            col=1,
        )

        metrics_strs.append(f"{name.split()[0]}: RMSE={rmse:.4e}")

    title_suffix = "<br>" + "<br>".join(metrics_strs)
    fig.update_layout(
        height=800,
        width=900,
        title={"text": f"Inverse Design: Flat-Top Spectrum {title_suffix}", "y": 0.96, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        margin={"t": 120},
    )
    fig.update_xaxes(title_text="Wavelength (µm)", row=2, col=1)
    fig.update_yaxes(title_text="|Amplitude| (Norm)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (rad)", row=2, col=1)
    fig.show()


def main() -> None:
    config = SimulationConfig()

    # 1. Physics & Bandwidth Conversion
    # d(Delta k)/d(lambda) approx for bandwidth conversion
    # Let's just calculate delta_k at bounds
    wl_center = config.design_wl
    print(f"Target Bandwidth: {config.target_bandwidth_nm} nm")

    # Convert nm to um for bandwidth calculation
    half_bw_um = (config.target_bandwidth_nm * 1e-3) / 2.0
    dk_plus = mgoslt.calc_twm_delta_k(wl_center + half_bw_um, wl_center + half_bw_um, config.design_temp)
    dk_minus = mgoslt.calc_twm_delta_k(wl_center - half_bw_um, wl_center - half_bw_um, config.design_temp)
    target_bandwidth_dk = abs(dk_plus - dk_minus).item()  # |dk| width

    print(f"Delta k width: {target_bandwidth_dk:.6f} / um")

    dk_val = mgoslt.calc_twm_delta_k(config.design_wl, config.design_wl, config.design_temp)
    Lp = 2 * (jnp.pi / dk_val)

    # Determine Sigmas
    # sigma_dk = factor * bandwidth
    # sigma_z = 1 / sigma_dk
    sigma_dk = config.smoothing_factor * target_bandwidth_dk
    sigma_z = 1.0 / sigma_dk
    print(f"Smoothing: sigma_dk={sigma_dk:.4e} /um, sigma_z={sigma_z:.4e} um")

    # 2. Generate Profile
    d_n, sign_profile = generate_sinc_profile(config.num_periods, Lp, target_bandwidth_dk, sigma_z)

    # 3. Pre-calculate Simulation space
    wls = jnp.linspace(config.wl_start, config.wl_end, config.wl_points)
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)

    # 4. Calculate Theoretical Target (Shape)
    target_shape = calculate_theoretical_spec_shape(config, wls, target_bandwidth_dk)

    # 5. Run Simulations
    cases: dict[str, Callable[[jax.Array, jax.Array, float, float], tuple[jax.Array, jax.Array]]] = {
        "Shifted Pulse Model": construct_geometry_shifted,
    }

    results: dict[str, tuple[jnp.ndarray, float]] = {}

    for name, constructor in cases.items():
        print(f"Simulating {name}...")
        w, k = constructor(d_n, sign_profile, Lp, config.kappa_mag)
        amps = run_simulation(w, k, dks)
        rmse = calculate_metrics(target_shape, amps)
        results[name] = (amps, rmse)
        print(f"Accuracy {name}: RMSE={rmse:.4e}")

    # 6. Prepare Plot Data
    # Normalize result to match target height 1.0 (since target is ideal unit rect)
    # Actually, inverse design usually scales to match pump power, but here we just check shape.
    # Let's normalize both to 1 for visual comparison.
    ref_amp, _ = results["Shifted Pulse Model"]
    peak_val = jnp.max(jnp.abs(ref_amp))

    # Scale result for plotting? No, typically normalize both to 1.
    # results stored raw amps. plot_results will take raw amps.
    # We should pass a target scaled to the peak of the simulation for visual overlay.

    scaled_targets = {"Target (Ideal)": target_shape * peak_val}

    plot_results(wls, results, scaled_targets)


if __name__ == "__main__":
    main()
