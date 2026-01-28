from dataclasses import dataclass

import jax
import jax.numpy as jnp
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

from qpm import cwes, mgoslt

jax.config.update("jax_enable_x64", val=True)


@dataclass
class SimulationConfig:
    """Configuration for the CPWPM SHG simulation."""

    # Physics
    design_wl: float = 1.064
    design_temp: float = 70.0
    kappa_mag: float = 1.5e-5 * (jnp.pi / 2)

    # Design
    total_length_um: float = 10000.0  # 10 mm

    # Spectral Target
    target_bandwidth_um: float = 0.1  # 20 nm in um

    # Simulation Wavelengths
    wl_start_um: float = 1.0
    wl_end_um: float = 1.14
    wl_points: int = 1000

    # Target Spectrum Parameters
    # Ratio of transition width (sigma) to flat-top width (bandwidth)
    apodization_sigma_ratio: float = 0.13
    # Fill factor defines the spatial chirp width relative to total length
    fill_factor: float = 0.4


def calculate_parameters(config: SimulationConfig) -> tuple[float, float]:
    """Calculates physical parameters."""
    wl_center = config.design_wl
    dk_center = mgoslt.calc_twm_delta_k(wl_center, wl_center, config.design_temp)

    wl_start = wl_center - config.target_bandwidth_um / 2
    wl_end = wl_center + config.target_bandwidth_um / 2

    dk_start = mgoslt.calc_twm_delta_k(wl_start, wl_start, config.design_temp)
    dk_end = mgoslt.calc_twm_delta_k(wl_end, wl_end, config.design_temp)

    dk_bandwidth = jnp.abs(dk_end - dk_start)

    return dk_center, dk_bandwidth


def generate_chirped_target(
    L_total: float, dk_center: float, dk_bandwidth: float, config: SimulationConfig
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """
    Generates the target spatial profile A(z) by IFT of a chirped frequency spectrum.

    Spectrum: Square Amplitude + Quadratic Phase.
    """
    # 1. Define Frequency Grid (centered at dk_center)
    num_points = 16384 * 2
    dk_range = 100 * dk_bandwidth

    dks = jnp.linspace(dk_center - dk_range / 2, dk_center + dk_range / 2, num_points)
    dk_rel = dks - dk_center

    # 2. Define Spectral Amplitude (Flat-Top)
    # Definition: Convolution of Rect(width=BW) and Gaussian(width=sigma)
    # This matches the profile used in pwpm_inverse_design.py

    sigma_freq = config.apodization_sigma_ratio * dk_bandwidth

    # Analytical convolution of Rect(width=BW) and Gaussian(sigma):
    # Result ~ erf((k + BW/2) / (sqrt(2)*sigma)) - erf((k - BW/2) / (sqrt(2)*sigma))

    arg_plus = (dk_rel + dk_bandwidth / 2.0) / (jnp.sqrt(2) * sigma_freq)
    arg_minus = (dk_rel - dk_bandwidth / 2.0) / (jnp.sqrt(2) * sigma_freq)

    amplitude_spectrum = 0.5 * (jax.scipy.special.erf(arg_plus) - jax.scipy.special.erf(arg_minus))

    # 3. Define Spectral Phase (Chirp)
    # Analytical determination of Chirp Rate (D2)
    fill_factor = config.fill_factor
    target_spatial_width = fill_factor * L_total
    D2 = target_spatial_width / dk_bandwidth

    phase_spectrum = 0.5 * D2 * dk_rel**2

    # 4. Construct Complex Spectrum
    S_k = amplitude_spectrum * jnp.exp(-1j * phase_spectrum)

    # 5. Inverse Fourier Transform to get A(z)
    A_z_full = jnp.fft.ifftshift(jnp.fft.ifft(jnp.fft.fftshift(S_k)))

    dK = dks[1] - dks[0]
    total_sim_len = 2 * jnp.pi / dK
    zs = jnp.linspace(-total_sim_len / 2, total_sim_len / 2, num_points)

    # Normalize
    A_z_norm = A_z_full / jnp.max(jnp.abs(A_z_full))

    # Crop to design L_total
    mask = jnp.abs(zs) <= L_total / 2
    zs_crop = zs[mask]
    A_z_crop = A_z_norm[mask]

    # Shift z to 0..L
    zs_final = zs_crop - zs_crop[0]

    # Check for truncation (Gibbs precursor)
    amp_start = jnp.abs(A_z_crop[0])
    amp_end = jnp.abs(A_z_crop[-1])
    amp_max = jnp.max(jnp.abs(A_z_crop))

    print("  Target Spatial Check:")
    print(f"    Peak Amplitude: {amp_max:.4f}")
    print(f"    Boundary Amplitudes: {amp_start:.4f} / {amp_end:.4f} ({(amp_start / amp_max) * 100:.1f}% / {(amp_end / amp_max) * 100:.1f}%)")
    if (amp_start / amp_max) > 0.01:
        print("    WARNING: Significant spatial truncation detected (>1%). This will cause spectral ripple.")

    return zs_final, A_z_crop, dk_center, D2


def design_chirped_grid(zs: jnp.ndarray, A_z: jnp.ndarray, dk_center: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Designs the Adaptive Grid Lambda(z) to track the phase of A(z).
    """
    phase_slow = jnp.unwrap(jnp.angle(A_z))
    dk_local_correction = jnp.gradient(phase_slow, zs)

    K_target = dk_center + dk_local_correction

    Phi_cumulative = jnp.cumsum(K_target) * (zs[1] - zs[0])
    Phi_cumulative -= Phi_cumulative[0]

    max_phase = Phi_cumulative[-1]
    n_max = int(max_phase / (2 * jnp.pi))
    ns = jnp.arange(n_max)
    target_phases = ns * 2 * jnp.pi

    interpolator = interp1d(Phi_cumulative, zs, kind="linear", bounds_error=False, fill_value="extrapolate")
    z_n_grid = interpolator(target_phases)

    L_n_grid = jnp.diff(z_n_grid, append=z_n_grid[-1] + (z_n_grid[-1] - z_n_grid[-2]))

    return z_n_grid, L_n_grid


def calculate_pwpm_parameters(
    z_grid: jnp.ndarray, L_grid: jnp.ndarray, zs_dense: jnp.ndarray, A_z_dense: jnp.ndarray, dk_center: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculates D_n and delta_n for each grid point.
    """
    mag_dense = jnp.abs(A_z_dense)
    interp_mag = interp1d(zs_dense, mag_dense, kind="linear", bounds_error=False, fill_value=0.0)
    mag_grid = interp_mag(z_grid)
    mag_grid = jnp.clip(mag_grid, 0.0, 1.0)

    D_n = jnp.arcsin(mag_grid) / jnp.pi

    interp_phase = interp1d(zs_dense, jnp.unwrap(jnp.angle(A_z_dense)), kind="linear", bounds_error=False, fill_value="extrapolate")
    phi_slow_grid = interp_phase(z_grid)

    phi_total_target = phi_slow_grid + dk_center * z_grid
    phi_grid_discrete = jnp.arange(len(z_grid)) * 2 * jnp.pi

    phi_resid = phi_total_target - phi_grid_discrete

    # Wrap phase
    phi_resid = (phi_resid + jnp.pi) % (2 * jnp.pi) - jnp.pi

    delta_n = phi_resid * L_grid / (2 * jnp.pi)

    # Check PWPM usage
    shift_ratio = jnp.abs(delta_n) / L_grid
    max_shift = jnp.max(shift_ratio)
    mean_shift = jnp.mean(shift_ratio)
    print(f"  PWPM Shift Check: Max={max_shift:.2%} Mean={mean_shift:.2%} (of period)")
    if max_shift < 0.01:
        print("    -> Dominant Mode: Chirp (Grid Adaptation)")
    else:
        print("    -> Dominant Mode: PWPM (Position Shift Active)")

    return D_n, delta_n


def construct_geometry(
    z_grid: jnp.ndarray, L_grid: jnp.ndarray, D_n: jnp.ndarray, delta_n: jnp.ndarray, kappa_mag: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Constructs the physical domain widths and kappas using vectorization.
    """
    z_c = z_grid + delta_n
    w = D_n * L_grid

    p_start = z_c - w / 2
    p_end = z_c + w / 2

    # Gap before first pulse
    z_sim_start = z_grid[0] - L_grid[0] / 2
    gap_0 = p_start[0] - z_sim_start

    # Gaps between pulses (Start[n] - End[n-1])
    gaps_mid = p_start[1:] - p_end[:-1]

    # Gap after last pulse
    z_sim_end = z_grid[-1] + L_grid[-1] / 2
    gap_last = z_sim_end - p_end[-1]

    # Combine all gaps: [Gap0, Gap1, ..., GapN]
    all_gaps = jnp.concatenate([jnp.array([gap_0]), gaps_mid, jnp.array([gap_last])])

    # Clamp to zero
    all_gaps = jnp.maximum(0.0, all_gaps)

    # Interleave Gaps and Pulses: [Gap0, Pulse0, Gap1, Pulse1, ..., GapN]
    N = len(w)
    total_segments = 2 * N + 1

    widths = jnp.zeros(total_segments)
    widths = widths.at[0::2].set(all_gaps)
    widths = widths.at[1::2].set(w)

    kappas = jnp.zeros(total_segments)
    kappas = kappas.at[0::2].set(kappa_mag)
    kappas = kappas.at[1::2].set(-kappa_mag)

    return widths, kappas


def run_simulation(widths: jnp.ndarray, kappas: jnp.ndarray, dks: jnp.ndarray) -> jnp.ndarray:
    """Runs the SHG simulation."""
    b_initial = jnp.array(1.0 + 0.0j)
    batch_simulate = jax.jit(jax.vmap(cwes.simulate_shg_npda, in_axes=(None, None, 0, None)))
    return batch_simulate(widths, kappas, dks, b_initial)


def find_bandwidth(wls: jnp.ndarray, efficiency: jnp.ndarray, threshold_ratio: float = 0.5) -> float:
    """Calculates the bandwidth."""
    max_eff = jnp.max(efficiency)
    threshold = max_eff * threshold_ratio
    above = efficiency > threshold
    wls_above = wls[above]
    if len(wls_above) == 0:
        return 0.0
    return wls_above[-1] - wls_above[0]


def main():
    config = SimulationConfig()
    print("Initializing Chirped PWPM Design...")

    dk_center, dk_bw_req = calculate_parameters(config)
    print(f"Center Wavelength: {config.design_wl} um")
    print(f"Target Bandwidth: {config.target_bandwidth_um:.4f} um")

    # 1. Generate Target Profile (Analytical Chirp)
    print("Generating Chirped Target...")
    # 1. Generate Target Profile (Analytical Chirp)
    print("Generating Chirped Target...")
    zs, A_z, _, D2 = generate_chirped_target(config.total_length_um, dk_center, dk_bw_req, config)
    print(f"Analytically Determined Chirp Rate (D2): {D2:.2e} um^2")

    # 2. Design Grid
    print("Designing Adaptive Grid...")
    z_grid, L_grid = design_chirped_grid(zs, A_z, dk_center)

    # 3. Calculate PWPM Parameters
    print("Calculating PWPM Parameters...")
    D_n, delta_n = calculate_pwpm_parameters(z_grid, L_grid, zs, A_z, dk_center)

    # 4. Construct Geometry
    widths, kappas = construct_geometry(z_grid, L_grid, D_n, delta_n, config.kappa_mag)
    print(f"Geometry constructed: {len(widths)} domains.")

    # 5. Simulate
    print("Simulating Spectrum...")
    wls = jnp.linspace(config.wl_start_um, config.wl_end_um, config.wl_points)
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)

    amps = run_simulation(widths, kappas, dks)
    efficiency = jnp.abs(amps) ** 2

    # 6. Analysis
    bw_fwhm_um = find_bandwidth(wls, efficiency, 0.5)
    bw_90_um = find_bandwidth(wls, efficiency, 0.9)

    print("-" * 30)
    print("RESULTS")
    print("-" * 30)
    print(f"FWHM Bandwidth: {bw_fwhm_um:.4f} um")
    print(f"0.5dB (90%) Bandwidth: {bw_90_um:.4f} um")
    print(f"Target Bandwidth: {config.target_bandwidth_um:.4f} um")
    print(f"Efficiency Peak: {jnp.max(efficiency):.4e}")

    # 7. Plotting
    fig = make_subplots(rows=2, cols=2, subplot_titles=("SHG Spectrum", "Target Profile A(z)", "Duty Cycle D(z)", "Local Period Lambda(z)"))

    # Spectrum
    fig.add_trace(go.Scatter(x=wls, y=efficiency, name="Efficiency"), row=1, col=1)
    fig.add_vline(x=config.design_wl - config.target_bandwidth_um / 2, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_vline(x=config.design_wl + config.target_bandwidth_um / 2, line_dash="dash", line_color="gray", row=1, col=1)

    # Target Profile (Abs)
    fig.add_trace(go.Scatter(x=zs, y=jnp.abs(A_z), name="|A(z)|"), row=1, col=2)

    # Duty Cycle
    fig.add_trace(go.Scatter(x=z_grid, y=D_n, name="D(z)", mode="markers+lines", marker_size=2), row=2, col=1)

    # Period
    fig.add_trace(go.Scatter(x=z_grid, y=L_grid, name="Lambda(z)", mode="markers+lines", marker_size=2), row=2, col=2)

    fig.update_layout(height=800, width=1200, title=f"Chirped PWPM (BW={config.target_bandwidth_um} um)")
    fig.update_xaxes(title_text="Wavelength (um)", row=1, col=1)
    fig.update_xaxes(title_text="Position (um)", row=1, col=2)
    fig.update_xaxes(title_text="Position (um)", row=2, col=1)
    fig.update_xaxes(title_text="Position (um)", row=2, col=2)

    fig.write_html("cpwpm_results.html")
    print("Plot saved to cpwpm_results.html")


if __name__ == "__main__":
    main()
