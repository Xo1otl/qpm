from dataclasses import dataclass

import jax
import jax.numpy as jnp
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from qpm import cwes, mgoslt

jax.config.update("jax_enable_x64", val=True)


@dataclass
class SimulationConfig:
    """Configuration for the SHG simulation."""

    # Physics
    design_wl: float = 1.064
    design_temp: float = 70.0
    kappa_mag: float = 1.5e-5 * (jnp.pi / 2)

    # Design
    total_length_um: float = 15000.0  # 15 mm
    target_bandwidth_um: float = 0.01
    apodization_sigma_ratio: float = 0.1  # Ratio of transition width to flat-top width

    # Simulation Wavelengths
    wl_start_um: float = 1.055
    wl_end_um: float = 1.075
    wl_points: int = 1000


def calculate_parameters(config: SimulationConfig) -> tuple[int, float, float, float]:
    """Calculates physical parameters."""
    wl_center = config.design_wl
    dk_center = mgoslt.calc_twm_delta_k(wl_center, wl_center, config.design_temp)

    # Poling period
    Lp = 2 * (jnp.pi / dk_center)

    # Calculate num_periods from total_length
    num_periods = int(config.total_length_um / Lp)

    # Bandwidth in delta_k
    wl_bw_start = wl_center - config.target_bandwidth_um / 2
    wl_bw_end = wl_center + config.target_bandwidth_um / 2
    dk_bw_start = mgoslt.calc_twm_delta_k(wl_bw_start, wl_bw_start, config.design_temp)
    dk_bw_end = mgoslt.calc_twm_delta_k(wl_bw_end, wl_bw_end, config.design_temp)
    dk_bandwidth = jnp.abs(dk_bw_end - dk_bw_start)

    return num_periods, Lp, dk_center, dk_bandwidth


def generate_target_profile(num_periods: int, Lp: float, dk_bandwidth: float, sigma_ratio: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generates the target spatial profile for a Flat-Top spectrum.

    Target Spectrum (Frequency Domain): Convolution of Rect(width=BW) and Gaussian(width=sigma).
    Target Profile (Spatial Domain): Product of Sinc(width=BW) and Gaussian(width=1/sigma).

    Returns:
        d_n: Duty cycle profile
        delta_n: Position shift profile
        z_n: Spatial coordinates
    """
    L_total = num_periods * Lp
    z_period_centers = (jnp.arange(num_periods) + 0.5) * Lp
    z_center = L_total / 2.0
    z_n = z_period_centers - z_center

    # Definition of widths
    # The spectral width is dk_bandwidth.
    # The target profile is A(z) ~ Sinc(dk_bandwidth * z / 2) * Gaussian(z)

    # 1. Main Sinc component (Fourier transform of Rect)
    # Rect(k / BW) <-> BW * Sinc(BW * z / 2) / (2pi) ... scaling factors can be normalized later
    arg_sinc = dk_bandwidth * z_n / 2.0
    profile_sinc = jnp.sinc(arg_sinc / jnp.pi)  # jnp.sinc is sin(pi*x)/(pi*x)

    # 2. Gaussian Apodization (Fourier transform of Gaussian convolution kernel)
    # A smaller sigma in frequency (sharper edges) means larger sigma in space (slower decay).
    # sigma_freq = sigma_ratio * dk_bandwidth
    # Gaussian(k) = exp(-k^2 / (2*sigma_freq^2)) <-> exp(-z^2 * sigma_freq^2 / 2)
    # sigma_ratio serves as a "smoothness" parameter.
    sigma_freq = sigma_ratio * dk_bandwidth
    profile_gauss = jnp.exp(-(z_n**2) * (sigma_freq**2) / 2.0)

    target_profile_complex = profile_sinc * profile_gauss

    # Normalize
    max_val = jnp.max(jnp.abs(target_profile_complex))
    norm_profile = target_profile_complex / max_val

    # Inverse Design (PWPM)
    # M(z) = |A(z)|
    # phi(z) = arg(A(z))
    # D(z) = arcsin(M(z)) / pi
    # delta(z) = phi(z) / Gm

    mag_profile = jnp.abs(norm_profile)
    # Clip to avoid numerical noise > 1.0 causing NaNs in arcsin
    mag_profile = jnp.clip(mag_profile, 0.0, 1.0)

    phase_profile = jnp.unwrap(jnp.angle(norm_profile))  # Should be 0 or pi for real target

    d_n = jnp.arcsin(mag_profile) / jnp.pi

    # Gm is the m=1 QPM grating vector component: 2*pi / Lp
    Gm = 2 * jnp.pi / Lp
    delta_n = phase_profile / Gm

    return d_n, delta_n, z_n


def construct_geometry_shifted(d_n: jnp.ndarray, delta_n: jnp.ndarray, Lp: float, kappa_mag: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Constructs the geometry using the Shifted Pulse Model."""

    num_periods = len(d_n)

    # Pulse width w_n = D_n * Lp
    w_n = d_n * Lp

    center_local = Lp / 2.0 + delta_n
    start_local = center_local - w_n / 2.0
    end_local = center_local + w_n / 2.0

    len1 = start_local
    len2 = w_n
    len3 = Lp - end_local

    widths = jnp.column_stack((len1, len2, len3)).ravel()

    # Signs
    # +1, -1, +1
    s1 = jnp.ones(num_periods)
    s2 = -jnp.ones(num_periods)
    s3 = jnp.ones(num_periods)
    signs = jnp.column_stack((s1, s2, s3)).ravel()

    kappas = kappa_mag * signs

    return widths, kappas


def run_simulation(widths: jnp.ndarray, kappas: jnp.ndarray, dks: jnp.ndarray) -> jnp.ndarray:
    """Runs the SHG simulation."""
    b_initial = jnp.array(1.0 + 0.0j)
    batch_simulate = jax.jit(jax.vmap(cwes.simulate_shg_npda, in_axes=(None, None, 0, None)))
    return batch_simulate(widths, kappas, dks, b_initial)


def plot_results(wls: jnp.ndarray, amps: jnp.ndarray, d_n: jnp.ndarray, delta_n: jnp.ndarray, z_n: jnp.ndarray, config: SimulationConfig) -> None:
    """Plots the simulation results and design profiles."""

    efficiency = jnp.abs(amps) ** 2
    phase = jnp.unwrap(jnp.angle(amps))

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("SHG Efficiency (Spectrum)", "Phase Response", "Duty Cycle Profile D(z)", "Shift Profile delta(z)"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # 1. Spectrum
    # x-axis in microns
    fig.add_trace(go.Scatter(x=wls, y=efficiency, mode="lines", name="Efficiency", line={"color": "#1f77b4"}), row=1, col=1)
    # Add bandwidth markers
    bw_half = config.target_bandwidth_um / 2
    fig.add_vline(x=config.design_wl - bw_half, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_vline(x=config.design_wl + bw_half, line_dash="dash", line_color="gray", row=1, col=1)

    # 2. Phase
    fig.add_trace(go.Scatter(x=wls, y=phase, mode="lines", name="Phase", line={"color": "#ff7f0e"}), row=1, col=2)

    # 3. Duty Cycle
    fig.add_trace(go.Scatter(x=z_n * 1e3, y=d_n, mode="lines", name="Duty Cycle", line={"color": "#2ca02c"}), row=2, col=1)

    # 4. Phase Shift
    fig.add_trace(go.Scatter(x=z_n * 1e3, y=delta_n * 1e6, mode="lines", name="Shift (um)", line={"color": "#d62728"}), row=2, col=2)

    fig.update_layout(
        height=900, width=1200, title_text=f"PWPM Inverse Design: Flat-Top {config.target_bandwidth_um * 1e3:.1f}nm", template="plotly_white"
    )

    fig.update_xaxes(title_text="Wavelength (um)", row=1, col=1)
    fig.update_xaxes(title_text="Wavelength (um)", row=1, col=2)
    fig.update_xaxes(title_text="Position (mm)", row=2, col=1)
    fig.update_xaxes(title_text="Position (mm)", row=2, col=2)

    fig.update_yaxes(title_text="Efficiency (a.u.)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (rad)", row=1, col=2)
    fig.update_yaxes(title_text="Duty Cycle [0-1]", row=2, col=1)
    fig.update_yaxes(title_text="Shift (um)", row=2, col=2)

    fig.write_html("pwpm_results.html")
    print("Results exported to pwpm_results.html")


def main() -> None:
    config = SimulationConfig()

    print("Initializing PWPM Design...")
    num_periods, Lp, dk_center, dk_bandwidth = calculate_parameters(config)
    print(f"Poling Period: {Lp:.4f} um")
    print(f"Calculated Number of Periods: {num_periods} (Length: {config.total_length_um} um)")
    print(f"Target Bandwidth (dk): {dk_bandwidth:.4E} rad/um")

    # 1. Generate Target
    d_n, delta_n, z_n = generate_target_profile(num_periods, Lp, dk_bandwidth, config.apodization_sigma_ratio)

    # 2. Construct Structure
    widths, kappas = construct_geometry_shifted(d_n, delta_n, Lp, config.kappa_mag)

    # 3. Simulate
    print("Simulating...")
    wls = jnp.linspace(config.wl_start_um, config.wl_end_um, config.wl_points)
    # wls is already in microns
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)

    amps = run_simulation(widths, kappas, dks)

    # 4. Plot
    plot_results(wls, amps, d_n, delta_n, z_n, config)


if __name__ == "__main__":
    main()
