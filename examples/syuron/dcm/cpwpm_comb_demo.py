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
    """Configuration for the CPWPM Comb simulation."""

    # Physics
    design_wl: float = 1.064
    design_temp: float = 70.0
    kappa_mag: float = 1.5e-5 * (jnp.pi / 2)

    # Design
    total_length_um: float = 15000.0

    # Comb Parameters
    comb_spacing_nm: float = 0.5
    num_modes: int = 15  # Total 2*N+1 modes
    mode_width_nm: float = 0.05  # Narrow bands

    # Simulation
    wl_start_um: float = 1.050
    wl_end_um: float = 1.080
    wl_points: int = 20000

    # Grid Factor (Oversampling for IFFT)
    ifft_padding: int = 16


def calculate_parameters(config: SimulationConfig):
    wl_center = config.design_wl
    dk_center = mgoslt.calc_twm_delta_k(wl_center, wl_center, config.design_temp)

    # Calculate dk spacing
    wl_spaced = wl_center + config.comb_spacing_nm * 1e-3
    dk_spaced = mgoslt.calc_twm_delta_k(wl_spaced, wl_spaced, config.design_temp)
    dk_spacing = jnp.abs(dk_spaced - dk_center)

    return dk_center, dk_spacing


def generate_comb_target(
    L_total: float, dk_center: float, dk_spacing: float, config: SimulationConfig
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generates a Comb spectrum target and its spatial IFFT."""

    # 1. Frequency Grid
    # We need high resolution to resolve the comb lines
    # And large range to resolve the narrow spatial features
    N_points = 16384 * config.ifft_padding

    # Range: needs to cover all modes with padding
    # Approx bandwidth
    total_bw_dk = config.num_modes * 2 * dk_spacing * 2.0
    dk_max_range = max(total_bw_dk, 200 * dk_spacing)

    dks = jnp.linspace(dk_center - dk_max_range / 2, dk_center + dk_max_range / 2, N_points)
    dk_rel = dks - dk_center

    # 2. Construct Spectrum (Sum of Gaussians)
    S_k = jnp.zeros_like(dks, dtype=jnp.complex128)

    # Convert mode width (nm) to dk
    wl_w = config.design_wl + config.mode_width_nm * 1e-3
    dk_w = mgoslt.calc_twm_delta_k(wl_w, wl_w, config.design_temp)
    sigma_dk = jnp.abs(dk_w - dk_center)

    # Add modes
    for m in range(-config.num_modes, config.num_modes + 1):
        center_k = m * dk_spacing
        gauss = jnp.exp(-((dk_rel - center_k) ** 2) / (2 * sigma_dk**2))
        # Optional: Add phase relationship between modes?
        # For a "Pulse Train" in space (Flat Comb), phases should be locked (e.g. 0)
        S_k += gauss

    # 3. IFFT to get A(z)
    A_z_full = jnp.fft.ifftshift(jnp.fft.ifft(jnp.fft.fftshift(S_k)))

    # Spatial Grid
    dK = dks[1] - dks[0]
    total_sim_len = 2 * jnp.pi / dK
    zs = jnp.linspace(-total_sim_len / 2, total_sim_len / 2, N_points)

    # Normalize
    A_z_norm = A_z_full / jnp.max(jnp.abs(A_z_full))

    # Crop to device length
    mask = jnp.abs(zs) <= L_total / 2
    zs_crop = zs[mask]
    A_z_crop = A_z_norm[mask]

    # Shift z to start at 0
    zs_final = zs_crop - zs_crop[0]

    return zs_final, A_z_crop, dk_center


def design_adaptive_grid(zs: jnp.ndarray, A_z: jnp.ndarray, dk_center: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Standard CPWPM adaptive grid logic.
    """
    phase_slow = jnp.unwrap(jnp.angle(A_z))

    # Total phase accumulation
    # phi_total(z) = K_center * z + phi_slow(z)
    phi_total = dk_center * zs + phase_slow
    phi_total -= phi_total[0]

    # Find positions where phase is 2*pi*n
    max_phase = phi_total[-1]
    n_max = int(max_phase / (2 * jnp.pi))
    ns = jnp.arange(n_max)
    target_phases = ns * 2 * jnp.pi

    interpolator = interp1d(phi_total, zs, kind="linear", bounds_error=False, fill_value="extrapolate")
    z_n = interpolator(target_phases)

    # Calculate local periods
    L_n = jnp.diff(z_n, append=z_n[-1] + (z_n[-1] - z_n[-2]))

    return z_n, L_n


def calculate_pwpm_parameters(
    z_grid: jnp.ndarray, L_grid: jnp.ndarray, zs_dense: jnp.ndarray, A_z_dense: jnp.ndarray, dk_center: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculates D_n and delta_n.
    """
    # Amplitude -> Duty Cycle
    mag_dense = jnp.abs(A_z_dense)
    interp_mag = interp1d(zs_dense, mag_dense, kind="linear", bounds_error=False, fill_value=0.0)
    mag_grid = interp_mag(z_grid)
    mag_grid = jnp.clip(mag_grid, 0.0, 1.0)

    D_n = jnp.arcsin(mag_grid) / jnp.pi

    # Phase -> Position Shift (delta)
    # Since grid is adaptive, this should be small, but it catches residual phase errors
    interp_phase = interp1d(zs_dense, jnp.unwrap(jnp.angle(A_z_dense)), kind="linear", bounds_error=False, fill_value="extrapolate")
    phi_slow_grid = interp_phase(z_grid)

    phi_total_target = phi_slow_grid + dk_center * z_grid
    phi_grid_discrete = jnp.arange(len(z_grid)) * 2 * jnp.pi

    phi_resid = phi_total_target - phi_grid_discrete
    phi_resid = (phi_resid + jnp.pi) % (2 * jnp.pi) - jnp.pi

    delta_n = phi_resid * L_grid / (2 * jnp.pi)

    # Check PWPM usage
    shift_ratio = jnp.abs(delta_n) / L_grid
    max_shift = jnp.max(shift_ratio)
    mean_shift = jnp.mean(shift_ratio)
    print(f"  PWPM Shift Check: Max={max_shift:.2%} Mean={mean_shift:.2%} (of period)")

    return D_n, delta_n


def construct_geometry(
    z_grid: jnp.ndarray, L_grid: jnp.ndarray, D_n: jnp.ndarray, delta_n: jnp.ndarray, kappa_mag: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    z_c = z_grid + delta_n
    w = D_n * L_grid

    p_start = z_c - w / 2
    p_end = z_c + w / 2

    # Build geometry vector
    # Segments: [Gap, Pulse, Gap, Pulse...]

    # Careful with overlaps!
    # For Comb (beating), D_n goes to 0 often. Overlaps shouldn't happen if grid is good.

    z_sim_start = z_grid[0] - L_grid[0] / 2
    gap_0 = p_start[0] - z_sim_start
    gaps_mid = p_start[1:] - p_end[:-1]
    z_sim_end = z_grid[-1] + L_grid[-1] / 2
    gap_last = z_sim_end - p_end[-1]

    all_gaps = jnp.concatenate([jnp.array([gap_0]), gaps_mid, jnp.array([gap_last])])
    all_gaps = jnp.maximum(0.0, all_gaps)  # clamp negative gaps (ovrlaps)

    total_segments = 2 * len(w) + 1
    widths = jnp.zeros(total_segments)
    widths = widths.at[0::2].set(all_gaps)
    widths = widths.at[1::2].set(w)

    kappas = jnp.zeros(total_segments)
    kappas = kappas.at[0::2].set(kappa_mag)
    kappas = kappas.at[1::2].set(-kappa_mag)

    return widths, kappas


def run_simulation(widths, kappas, dks):
    b_initial = jnp.array(1.0 + 0.0j)
    batch_simulate = jax.jit(jax.vmap(cwes.simulate_shg_npda, in_axes=(None, None, 0, None)))
    return batch_simulate(widths, kappas, dks, b_initial)


def main():
    config = SimulationConfig()
    print("Initializing Universal CPWPM Comb Demo...")

    dk_center, dk_spacing = calculate_parameters(config)
    print(f"Targeting {2 * config.num_modes + 1} modes with spacing {config.comb_spacing_nm} nm")

    # 1. Target
    zs, A_z, _ = generate_comb_target(config.total_length_um, dk_center, dk_spacing, config)

    # 2. Grid
    z_grid, L_grid = design_adaptive_grid(zs, A_z, dk_center)

    # 3. Parameters
    D_n, delta_n = calculate_pwpm_parameters(z_grid, L_grid, zs, A_z, dk_center)

    # 4. Geometry
    widths, kappas = construct_geometry(z_grid, L_grid, D_n, delta_n, config.kappa_mag)
    print(f"Constructed {len(widths) // 2} domains")

    # 5. Simulate
    print("Simulating...")
    wls = jnp.linspace(config.wl_start_um, config.wl_end_um, config.wl_points)
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)
    amps = run_simulation(widths, kappas, dks)

    # 6. Plot
    eff = jnp.abs(amps) ** 2

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Spectrum", "Target |A(z)|", "Duty Cycle D(z)", "Grid Deviation (PWPM)"))

    fig.add_trace(go.Scatter(x=wls, y=eff, name="Efficiency"), row=1, col=1)
    fig.add_trace(go.Scatter(x=zs, y=jnp.abs(A_z), name="|A(z)|"), row=1, col=2)
    fig.add_trace(go.Scatter(x=z_grid, y=D_n, name="D(z)", mode="lines"), row=2, col=1)

    # Plot delta_n to see if PWPM is active
    fig.add_trace(go.Scatter(x=z_grid, y=delta_n, name="Shift delta(z)", mode="lines"), row=2, col=2)

    fig.update_layout(height=800, width=1200, title="Universal CPWPM: Flat Comb Target")
    fig.write_html("cpwpm_comb_result.html")
    print("Saved cpwpm_comb_result.html")


if __name__ == "__main__":
    main()
