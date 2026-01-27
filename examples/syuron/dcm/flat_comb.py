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

    num_periods: int = 20000
    design_wl: float = 1.064
    design_temp: float = 70.0
    wl_start: float = 1.02
    wl_end: float = 1.108
    wl_points: int = 100000
    kappa_mag: float = 1.5e-5 / (2 / jnp.pi)

    # Comb specific
    comb_modes: int = 50  # Number of sidebands on each side (Total 2*M + 1)
    comb_spacing_nm: float = 0.5  # Spacing between modes in nm


def generate_gaussian_comb_profile(
    num_periods: int,
    Lp: float,
    num_modes: int,
    spacing_dk: float,
    sigma_ratio: float = 4.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    L_total = num_periods * Lp
    z_period_centers = (jnp.arange(num_periods) + 0.5) * Lp
    z_center = L_total / 2.0
    z_n = z_period_centers - z_center

    N = 2 * num_modes + 1
    arg_num = N * spacing_dk * z_n / 2.0
    arg_den = spacing_dk * z_n / 2.0

    denom = jnp.sin(arg_den)
    numerator = jnp.sin(arg_num)
    comb_oscillation = jnp.where(jnp.abs(denom) < 1e-10, N, numerator / denom)

    spatial_sigma = L_total / sigma_ratio
    gaussian_window = jnp.exp(-(z_n**2) / (2 * spatial_sigma**2))

    target_profile = comb_oscillation * gaussian_window

    max_val = jnp.max(jnp.abs(target_profile))
    norm_profile = target_profile / max_val

    mag_profile = jnp.abs(norm_profile)
    sign_profile = jnp.sign(norm_profile)
    d_n = jnp.arcsin(mag_profile) / jnp.pi

    return d_n, sign_profile


def construct_geometry_shifted(d_n: jnp.ndarray, sign_profile: jnp.ndarray, Lp: float, kappa_mag: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Constructs the geometry using the Shifted Pulse Model."""
    is_pos = sign_profile > 0
    w1 = jnp.where(is_pos, (1 - d_n) * Lp / 2.0, d_n * Lp / 2.0)
    w2 = Lp - 2 * w1
    w3 = w1

    widths = jnp.column_stack((w1, w2, w3)).ravel()

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


def plot_results(wls: jnp.ndarray, amps: jnp.ndarray, config: SimulationConfig) -> None:
    """Plots the simulation results."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Amplitude Spectrum", "Phase Response"))

    # Engineered
    fig.add_trace(
        go.Scatter(x=wls, y=jnp.abs(amps), mode="lines", name="Comb Output", line={"color": "#1f77b4", "width": 2}),
        row=1,
        col=1,
    )

    # Phase
    phi = jnp.unwrap(jnp.angle(amps))
    fig.add_trace(
        go.Scatter(x=wls, y=phi, mode="lines", name="Phase", line={"color": "#1f77b4", "width": 1.5}, showlegend=False),
        row=2,
        col=1,
    )

    # Layout
    fig.update_layout(
        height=800,
        width=900,
        title={"text": f"DCM Flat-Comb: {2 * config.comb_modes + 1} Modes, {config.comb_spacing_nm}nm Spacing", "x": 0.5},
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Wavelength (µm)", row=2, col=1)
    fig.update_yaxes(title_text="|Amplitude|", row=1, col=1)

    fig.write_html("flat_comb_results.html")
    print("Results exported to flat_comb_results.html")


def main() -> None:
    config = SimulationConfig()

    # 1. Physics Setup
    wl_center = config.design_wl

    # Calculate dk spacing
    dk_center = mgoslt.calc_twm_delta_k(wl_center, wl_center, config.design_temp)
    dk_plus = mgoslt.calc_twm_delta_k(wl_center + config.comb_spacing_nm * 1e-3, wl_center + config.comb_spacing_nm * 1e-3, config.design_temp)
    spacing_dk = abs(dk_plus - dk_center).item()

    print(f"Comb Spacing: {config.comb_spacing_nm} nm -> {spacing_dk:.4f} rad/um")

    Lp = 2 * (jnp.pi / dk_center)

    # 2. Design
    d_n, sign_profile = generate_gaussian_comb_profile(config.num_periods, Lp, config.comb_modes, spacing_dk)

    # 3. Simulation Space
    wls = jnp.linspace(config.wl_start, config.wl_end, config.wl_points)
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)

    # 4. Run
    print("Simulating...")
    w, k = construct_geometry_shifted(d_n, sign_profile, Lp, config.kappa_mag)
    amps = run_simulation(w, k, dks)

    # 5. Plot
    plot_results(wls, amps, config)


if __name__ == "__main__":
    main()
