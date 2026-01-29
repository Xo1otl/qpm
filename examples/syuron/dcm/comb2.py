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
    # QPM period roughly ~30um for 1064 SHG in LN, kappa is standard
    kappa_mag: float = 1.5e-5 * (jnp.pi / 2)

    # Design
    total_length_um: float = 20000.0  # Slightly longer to accommodate chirped pulse

    # Comb Parameters
    comb_spacing_nm: float = 0.5
    num_modes: int = 20  # Total 2*N+1 modes
    mode_width_nm: float = 0.08

    # --- Phase Engineering (Crucial for Manufacturability) ---
    # 分散を与えて空間的なパルスを広げる (Talbot効果/チャープの応用)
    # 値が大きいほどDuty比の変動が緩やかになり、ピークが下がる
    dispersion_factor: float = 2.5

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
    """Generates a Comb spectrum target and its spatial IFFT with Phase Engineering."""

    N_points = 16384 * config.ifft_padding

    # Bandwidth calculation
    total_bw_dk = config.num_modes * 2 * dk_spacing * 2.0
    dk_max_range = max(total_bw_dk, 200 * dk_spacing)

    dks = jnp.linspace(dk_center - dk_max_range / 2, dk_center + dk_max_range / 2, N_points)
    dk_rel = dks - dk_center

    # --- 2. Construct Spectrum with Phase Engineering ---
    S_k = jnp.zeros_like(dks, dtype=jnp.complex128)

    # Convert mode width
    wl_w = config.design_wl + config.mode_width_nm * 1e-3
    dk_w = mgoslt.calc_twm_delta_k(wl_w, wl_w, config.design_temp)
    sigma_dk = jnp.abs(dk_w - dk_center)

    # Add modes with Quadratic Phase (Parabolic Phase)
    # phi_m = gamma * m^2
    # これがないと空間上でデルタ関数になり、Duty比が破綻する

    for m in range(-config.num_modes, config.num_modes + 1):
        center_k = m * dk_spacing

        # Amplitude (Gaussian)
        ampl = jnp.exp(-((dk_rel - center_k) ** 2) / (2 * sigma_dk**2))

        # Phase (Quadratic / Chirp)
        # 各モードに2次の位相を与えることで、空間的な干渉を分散させる
        phase_val = config.dispersion_factor * (m**2)

        S_k += ampl * jnp.exp(1j * phase_val)

    # 3. IFFT to get A(z)
    A_z_full = jnp.fft.ifftshift(jnp.fft.ifft(jnp.fft.fftshift(S_k)))

    # Spatial Grid
    dK = dks[1] - dks[0]
    total_sim_len = 2 * jnp.pi / dK
    zs = jnp.linspace(-total_sim_len / 2, total_sim_len / 2, N_points)

    # Normalize
    # 位相分散を入れたことで、ピーク値が下がり、エネルギーが広がるため
    # 正規化後のプロファイルが「太く」なり、Duty比が稼げるようになる
    A_z_norm = A_z_full / jnp.max(jnp.abs(A_z_full))

    # Crop to device length
    mask = jnp.abs(zs) <= L_total / 2
    zs_crop = zs[mask]
    A_z_crop = A_z_norm[mask]

    # Shift z to start at 0
    zs_final = zs_crop - zs_crop[0]

    return zs_final, A_z_crop, dk_center


def design_adaptive_grid(zs: jnp.ndarray, A_z: jnp.ndarray, dk_center: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Standard CPWPM adaptive grid logic."""
    phase_slow = jnp.unwrap(jnp.angle(A_z))

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
    # Amplitude -> Duty Cycle
    mag_dense = jnp.abs(A_z_dense)
    interp_mag = interp1d(zs_dense, mag_dense, kind="linear", bounds_error=False, fill_value=0.0)
    mag_grid = interp_mag(z_grid)

    # Clip magnitude to avoid math errors, though max should be 1.0
    mag_grid = jnp.clip(mag_grid, 0.0, 1.0)
    D_n = jnp.arcsin(mag_grid) / jnp.pi

    # Phase -> Position Shift (delta)
    interp_phase = interp1d(zs_dense, jnp.unwrap(jnp.angle(A_z_dense)), kind="linear", bounds_error=False, fill_value="extrapolate")
    phi_slow_grid = interp_phase(z_grid)

    phi_total_target = phi_slow_grid + dk_center * z_grid
    phi_grid_discrete = jnp.arange(len(z_grid)) * 2 * jnp.pi

    phi_resid = phi_total_target - phi_grid_discrete
    phi_resid = (phi_resid + jnp.pi) % (2 * jnp.pi) - jnp.pi

    delta_n = phi_resid * L_grid / (2 * jnp.pi)

    return D_n, delta_n


def construct_geometry(
    z_grid: jnp.ndarray, L_grid: jnp.ndarray, D_n: jnp.ndarray, delta_n: jnp.ndarray, kappa_mag: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    z_c = z_grid + delta_n
    w = D_n * L_grid

    p_start = z_c - w / 2
    p_end = z_c + w / 2

    z_sim_start = z_grid[0] - L_grid[0] / 2
    gap_0 = p_start[0] - z_sim_start
    gaps_mid = p_start[1:] - p_end[:-1]
    z_sim_end = z_grid[-1] + L_grid[-1] / 2
    gap_last = z_sim_end - p_end[-1]

    all_gaps = jnp.concatenate([jnp.array([gap_0]), gaps_mid, jnp.array([gap_last])])
    all_gaps = jnp.maximum(0.0, all_gaps)

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
    print("Initializing CPWPM Comb Demo with Phase Engineering...")

    dk_center, dk_spacing = calculate_parameters(config)
    print(f"Targeting {2 * config.num_modes + 1} modes with spacing {config.comb_spacing_nm} nm")

    # 1. Target Generation (Now with Dispersion!)
    zs, A_z, _ = generate_comb_target(config.total_length_um, dk_center, dk_spacing, config)

    # 2. Grid Design
    z_grid, L_grid = design_adaptive_grid(zs, A_z, dk_center)

    # 3. Parameter Calculation
    D_n, delta_n = calculate_pwpm_parameters(z_grid, L_grid, zs, A_z, dk_center)

    # Stats
    print(f"Duty Ratio Stats: Mean={jnp.mean(D_n):.3f}, Max={jnp.max(D_n):.3f}")
    if jnp.mean(D_n) < 0.01:
        print("WARNING: Mean Duty Ratio is very low. Increase 'dispersion_factor'.")

    # 4. Geometry Construction
    widths, kappas = construct_geometry(z_grid, L_grid, D_n, delta_n, config.kappa_mag)
    print(f"Constructed {len(widths) // 2} domains")

    # 5. Simulation
    print("Simulating...")
    wls = jnp.linspace(config.wl_start_um, config.wl_end_um, config.wl_points)
    dks = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)
    amps = run_simulation(widths, kappas, dks)

    # 6. Plotting
    eff = jnp.abs(amps) ** 2

    # Layout: 3 Rows
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        subplot_titles=("Output Spectrum (Efficiency)", "Spatial Target Amplitude |A(z)|", "Duty Ratio Profile D(z)"),
        vertical_spacing=0.1,
    )

    # Row 1: Spectrum
    fig.add_trace(go.Scatter(x=wls, y=eff, name="Efficiency", line=dict(color="blue")), row=1, col=1)
    fig.update_xaxes(title_text="Wavelength (um)", row=1, col=1)

    # Row 2: Spatial Amplitude
    fig.add_trace(go.Scatter(x=zs, y=jnp.abs(A_z), name="|A(z)| Target", line=dict(color="green")), row=2, col=1)
    fig.update_xaxes(title_text="Position z (um)", row=2, col=1)

    # Row 3: Duty Ratio (The requested plot)
    fig.add_trace(
        go.Scatter(x=z_grid, y=D_n, name="Duty Ratio D(z)", mode="lines+markers", marker=dict(size=2), line=dict(color="red", width=1)), row=3, col=1
    )
    fig.update_yaxes(title_text="Duty Ratio (0-0.5)", range=[0, 0.6], row=3, col=1)
    fig.update_xaxes(title_text="Position z (um)", row=3, col=1)

    fig.update_layout(height=1000, width=900, title=f"CPWPM Comb: {2 * config.num_modes + 1} modes, Dispersion={config.dispersion_factor}")
    fig.write_html("cpwpm_comb_result.html")
    print("Saved cpwpm_comb_result.html")


if __name__ == "__main__":
    main()
