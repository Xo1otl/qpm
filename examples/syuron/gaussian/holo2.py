import matplotlib.pyplot as plt
import numpy as np

# --- 1. Core Physics & Functions ---
SELLMEIER_PARAMS = {
    "a": np.array([4.5615, 0.08488, 0.1927, 5.5832, 8.3067, 0.021696]),
    "b": np.array([4.782e-07, 3.0913e-08, 2.7326e-08, 1.4837e-05, 1.3647e-07]),
}


def sellmeier_n_eff(wl: np.ndarray, temp: float) -> np.ndarray:
    f = (temp - 24.5) * (temp + 24.5 + 2.0 * 273.16)
    lambda_sq = wl**2
    a, b = SELLMEIER_PARAMS["a"], SELLMEIER_PARAMS["b"]
    n_sq = (
        a[0]
        + b[0] * f
        + (a[1] + b[1] * f) / (lambda_sq - (a[2] + b[2] * f) ** 2)
        + (a[3] + b[3] * f) / (lambda_sq - (a[4] + b[4] * f) ** 2)
        - a[5] * lambda_sq
    )
    return np.sqrt(n_sq)


def calc_twm_delta_k(wl1: np.ndarray, wl2: np.ndarray, t: float) -> np.ndarray:
    wl_sum = (wl1 * wl2) / (wl1 + wl2)
    n1 = sellmeier_n_eff(wl1, t)
    n2 = sellmeier_n_eff(wl2, t)
    n_sum = sellmeier_n_eff(wl_sum, t)
    return 2.0 * np.pi * (n_sum / wl_sum - n1 / wl1 - n2 / wl2)


def calculate_local_shg_amplitudes(domain_widths, kappa_vals, delta_k, b_initial):
    gamma = delta_k / 2.0
    a_omega_sq = b_initial**2
    gamma_l = gamma * domain_widths
    sinc_term = np.sinc(gamma_l / np.pi)
    return -1j * kappa_vals * a_omega_sq * domain_widths * np.exp(1j * gamma_l) * sinc_term


def simulate_shg_npda(domain_widths, kappa_vals, delta_k, b_initial):
    dk_col = delta_k[:, np.newaxis]
    w_row = domain_widths[np.newaxis, :]
    k_row = kappa_vals[np.newaxis, :]
    local_amplitudes = calculate_local_shg_amplitudes(w_row, k_row, dk_col, b_initial)
    z_starts = np.concatenate([np.array([0.0]), np.cumsum(domain_widths[:-1])])
    z_starts_row = z_starts[np.newaxis, :]
    phase_factors = np.exp(1j * dk_col * z_starts_row)
    return np.sum(local_amplitudes * phase_factors, axis=1)


def merge_and_filter_domains(widths, kappas, threshold=1.0e-6):
    widths = np.array(widths)
    kappas = np.array(kappas)
    if len(widths) == 0:
        return np.array([]), np.array([])
    is_change = np.concatenate((kappas[:-1] != kappas[1:], np.array([True])))
    end_indices = np.where(is_change)[0]
    cum_widths = np.cumsum(widths)
    boundaries = cum_widths[end_indices]
    merged_widths = np.diff(np.concatenate((np.array([0.0]), boundaries)))
    merged_kappas = kappas[end_indices]

    w_list = merged_widths.tolist()
    k_list = merged_kappas.tolist()
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(w_list):
            if w_list[i] < threshold:
                if i == 0:
                    if len(w_list) > 1:
                        w_list[1] += w_list[i]
                        w_list.pop(0)
                        k_list.pop(0)
                        changed = True
                        continue
                    else:
                        i += 1
                elif i == len(w_list) - 1:
                    w_list[i - 1] += w_list[i]
                    w_list.pop(i)
                    k_list.pop(i)
                    changed = True
                    continue
                else:
                    w_list[i - 1] += w_list[i] + w_list[i + 1]
                    w_list.pop(i)
                    w_list.pop(i)
                    k_list.pop(i)
                    k_list.pop(i)
                    changed = True
                    continue
            i += 1
    return np.array(w_list), np.array(k_list)


# --- 2. Main Logic ---
def main():
    # --- Configuration ---
    num_periods = 2000
    design_wl = 1.064
    design_temp = 70.0
    kappa_mag = 1.31e-5 / (2 / np.pi)

    # Use sigma = L/4 as discussed
    spatial_sigma_ratio = 4.0

    # Physics parameters
    dk_design = calc_twm_delta_k(np.array(design_wl), np.array(design_wl), design_temp)
    Lp = 2 * (np.pi / dk_design)
    Lc = Lp / 2

    print(f"Lp: {Lp:.4f} um, Lc: {Lc:.4f} um")

    # --- Design Setup ---
    L_total = num_periods * Lp
    z_period_centers = (np.arange(num_periods) + 0.5) * Lp
    z_n = z_period_centers - L_total / 2.0
    spatial_sigma = L_total / spatial_sigma_ratio

    # Spatial Gaussian Profile
    target_profile = np.exp(-(z_n**2) / (2 * (spatial_sigma**2)))
    d_n = np.arcsin(np.abs(target_profile)) / np.pi
    sign_profile = np.sign(target_profile)  # All positive for simple Gaussian

    # --- Dithered Construction ---
    dx_um = 0.01
    num_steps = int(np.ceil(Lp / dx_um))
    possible_widths = np.arange(num_steps + 1) * dx_um
    possible_duties = possible_widths / Lp
    possible_effs = np.sin(np.pi * possible_duties)
    target_effs = np.sin(np.pi * d_n)

    d_dithered = np.zeros_like(target_effs)
    accum_error = 0.0
    for i, target in enumerate(target_effs):
        desired = target + accum_error
        diffs = np.abs(possible_effs - desired)
        path_idx = np.argmin(diffs)
        d_dithered[i] = possible_duties[path_idx]
        accum_error = desired - possible_effs[path_idx]

    gap_widths = (1 - d_dithered) * Lp / 2.0
    pulse_widths = d_dithered * Lp
    widths = np.column_stack((gap_widths, pulse_widths, gap_widths)).ravel()
    base_signs = np.tile(np.array([1.0, -1.0, 1.0]), num_periods)
    kappas = kappa_mag * base_signs * np.repeat(sign_profile, 3)

    final_widths, final_kappas = merge_and_filter_domains(widths, kappas, threshold=1e-2)

    # --- Simulation ---
    wls = np.linspace(design_wl - 0.003, design_wl + 0.003, 1000)
    dks = calc_twm_delta_k(wls, wls, design_temp)
    b_initial = 1.0 + 0.0j

    # 1. Finite Dithered Simulation
    amps_sim = simulate_shg_npda(final_widths, final_kappas, dks, b_initial)
    spec_sim = np.abs(amps_sim)

    # 2. Infinite Pure Gaussian (Analytical)
    # The spectrum of Gaussian(z) is Gaussian(delta_k)
    # Center of band is where delta_k = dk_design
    # Phase mismatch relative to QPM: delta_beta = dks - dk_design
    delta_beta = dks - dk_design

    # Analytical Fourier Transform of exp(-z^2 / 2sigma^2) is proportional to exp(-k^2 sigma^2 / 2)
    spec_infinite = np.exp(-(delta_beta**2 * spatial_sigma**2) / 2.0)

    # Normalize for comparison
    # We normalize both to 1.0 to compare the SHAPE (ripples vs smooth)
    spec_sim_norm = spec_sim / np.max(spec_sim)
    spec_infinite_norm = spec_infinite / np.max(spec_infinite)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(wls, spec_infinite_norm, "r--", linewidth=1.5, label="Infinite Pure Gaussian (Analytical)")
    plt.plot(wls, spec_sim_norm, "b-", alpha=0.7, label=f"Finite Dithered Simulation (L={L_total / 1000:.1f}mm)")

    plt.title("Spectral Comparison: Finite vs Infinite Gaussian")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Normalized SHG Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("holo2.png")


if __name__ == "__main__":
    main()
