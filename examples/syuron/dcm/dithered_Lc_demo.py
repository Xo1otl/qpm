import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from qpm import cwes, mgoslt
import numpy as np

jax.config.update("jax_enable_x64", val=True)


def merge_and_filter_domains(widths, kappas, threshold=1.0e-6):
    """
    Merges adjacent same-sign domains and filters small ones.
    Uses vectorized operations for the initial merge (RLE).
    """
    # ensure inputs are jax/numpy arrays
    widths = jnp.array(widths)
    kappas = jnp.array(kappas)

    if len(widths) == 0:
        return jnp.array([]), jnp.array([])

    # 1. Fast Vectorized Merge (Run-Length Encoding)
    # Identify indices where kappa changes or it's the end of the array
    # kappas[:-1] != kappas[1:] creates a boolean mask of changes
    is_change = jnp.concatenate((kappas[:-1] != kappas[1:], jnp.array([True])))

    # Indices where a domain ENDS
    end_indices = jnp.where(is_change)[0]

    # Cumulative sum of widths allows us to calculating merged widths by differencing
    cum_widths = jnp.cumsum(widths)
    boundaries = cum_widths[end_indices]

    # Calculate merged widths
    # Prepend 0 to boundaries to compute diffs
    merged_widths = jnp.diff(jnp.concatenate((jnp.array([0.0]), boundaries)))

    # Calculate merged kappas (just take the value at the end index)
    merged_kappas = kappas[end_indices]

    print(f"Initial Reduce: {len(widths)} -> {len(merged_widths)} domains")

    # 2. Filter small domains
    # Since the number of domains is now small (~2000 vs 30000),
    # a Python loop is efficient enough and easier to handle the
    # complex logic of deleting and merging neighbors.

    w_list = np.array(merged_widths).tolist()
    k_list = np.array(merged_kappas).tolist()

    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        i = 0
        while i < len(w_list):
            if w_list[i] < threshold:
                # Need to merge w_list[i] into neighbors
                # Case 1: Start of array
                if i == 0:
                    if len(w_list) > 1:
                        # [small, Big, ...] -> [Big+small, ...]
                        # effectively small becomes part of Big's domain
                        w_list[1] += w_list[i]
                        w_list.pop(0)
                        k_list.pop(0)
                        changed = True
                        continue
                    else:
                        # Only one domain left and it's small? just keep it or remove?
                        # If we remove, we have nothing. Keep it.
                        i += 1
                # Case 2: End of array
                elif i == len(w_list) - 1:
                    # [..., Big, small] -> [..., Big+small]
                    w_list[i - 1] += w_list[i]
                    w_list.pop(i)
                    k_list.pop(i)
                    changed = True
                    continue
                # Case 3: Middle
                else:
                    # [Left, small, Right]
                    # Since we merged same-signs in step 1, Left and Right MUST have
                    # opposite sign to small. Thus Left and Right have SAME sign.
                    # We merge [Left, small, Right] -> [Left + small + Right]
                    w_list[i - 1] += w_list[i] + w_list[i + 1]
                    w_list.pop(i)  # remove small
                    w_list.pop(i)  # remove Right (which is now at i)
                    k_list.pop(i)  # remove small kappa
                    k_list.pop(i)  # remove Right kappa
                    changed = True
                    continue
            i += 1

    return jnp.array(w_list), jnp.array(k_list)


def main():
    # --- Configuration ---
    num_periods = 10000
    design_wl = 1.064
    design_temp = 70.0
    kappa_mag = 1.31e-5 / (2 / jnp.pi)
    spatial_sigma_ratio = 8.0

    # Physics parameters
    dk_val = mgoslt.calc_twm_delta_k(design_wl, design_wl, design_temp)
    Lp = 2 * (jnp.pi / dk_val)
    Lc = Lp / 2
    res_nm = Lc * 1000
    dx_um = res_nm / 1000.0

    print(f"Lc: {Lc:.4f} um, Resolution: {res_nm:.2f} nm")

    # --- 1. Inverse Design (Gaussian) ---
    L_total = num_periods * Lp
    z_period_centers = (jnp.arange(num_periods) + 0.5) * Lp
    z_n = z_period_centers - L_total / 2.0
    spatial_sigma = L_total / spatial_sigma_ratio
    target_profile = jnp.exp(-(z_n**2) / (2 * (spatial_sigma**2)))
    norm_profile = target_profile / jnp.max(jnp.abs(target_profile))
    d_n = jnp.arcsin(jnp.abs(norm_profile)) / jnp.pi
    sign_profile = jnp.sign(norm_profile)

    # --- 2. Quantization (Error Diffusion) ---
    num_steps = int(jnp.ceil(Lp / dx_um))
    possible_widths = jnp.arange(num_steps + 1) * dx_um
    possible_widths = possible_widths[possible_widths <= Lp + 1e-9]
    possible_duties = possible_widths / Lp
    possible_effs = jnp.sin(jnp.pi * possible_duties)
    target_effs = jnp.sin(jnp.pi * d_n)

    def scan_body(carry, target):
        accum_error = carry
        desired = target + accum_error
        diffs = jnp.abs(possible_effs - desired)
        path_idx = jnp.argmin(diffs)
        chosen_eff = possible_effs[path_idx]
        chosen_duty = possible_duties[path_idx]
        new_error = desired - chosen_eff
        return new_error, chosen_duty

    _, d_dithered = jax.lax.scan(scan_body, 0.0, target_effs)

    # Construct Initial Geometry
    gap_widths = (1 - d_dithered) * Lp / 2.0
    pulse_widths = d_dithered * Lp
    widths = jnp.column_stack((gap_widths, pulse_widths, gap_widths)).ravel()

    base_signs = jnp.tile(jnp.array([1.0, -1.0, 1.0]), num_periods)
    period_signs = jnp.repeat(sign_profile, 3)
    kappas = kappa_mag * base_signs * period_signs

    # --- 2b. Merge and Filter ---
    print("Merging and filtering domains...")
    final_widths, final_kappas = merge_and_filter_domains(widths, kappas, threshold=1e-2)
    print(f"Original segments: {len(widths)}, Final domains: {len(final_widths)}")
    if len(final_widths) > 0:
        print(f"Min width after filtering: {jnp.min(final_widths):.4e} um")

    # --- 3. Simulation ---
    wl_start, wl_end, wl_points = 1.0638, 1.0642, 1000
    wls = jnp.linspace(wl_start, wl_end, wl_points)
    dks = mgoslt.calc_twm_delta_k(wls, wls, design_temp)
    b_initial = jnp.array(1.0 + 0.0j)

    print("Running simulation...")
    batch_simulate = jax.jit(jax.vmap(cwes.simulate_shg_npda, in_axes=(None, None, 0, None)))
    amps = batch_simulate(final_widths, final_kappas, dks, b_initial)
    amps.block_until_ready()
    spectrum = jnp.abs(amps)

    # Calculate Target Spectrum
    print("Simulating Ideal Target...")
    d_ideal = d_n
    gap_ideal = (1 - d_ideal) * Lp / 2.0
    pulse_ideal = d_ideal * Lp
    w_ideal = jnp.column_stack((gap_ideal, pulse_ideal, gap_ideal)).ravel()
    amps_ideal = batch_simulate(w_ideal, kappas, dks, b_initial)
    spectrum_ideal = jnp.abs(amps_ideal)

    # --- 4. Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Spectra
    ax1.plot(wls, spectrum_ideal, "k--", label="Target (Ideal)", linewidth=1.5)
    ax1.plot(wls, spectrum, "g-", label="Dithered & Filtered (Lc)", linewidth=1.0)
    ax1.set_title("Spectral Distribution (After Merging/Filtering)")
    ax1.set_xlabel("Wavelength (µm)")
    ax1.set_ylabel("SHG Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Domain Width Distribution (Merged)
    domain_indices = jnp.arange(len(final_widths))

    # Plot ALL domains
    ax2.scatter(domain_indices, final_widths, s=1, c="b", alpha=0.5, label="Domain Widths")

    # Add horizontal lines
    ax2.axhline(y=Lc, color="gray", linestyle=":", alpha=0.3, label="Lc")
    ax2.axhline(y=2 * Lc, color="gray", linestyle=":", alpha=0.3)
    ax2.axhline(y=3 * Lc, color="gray", linestyle=":", alpha=0.3)

    ax2.set_title("Domain Width Distribution (Merged & Filtered)")
    ax2.set_xlabel("Domain Index")
    ax2.set_ylabel("Domain Width (µm)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("dithered_Lc_demo.png", dpi=150)
    print("Saved dithered_Lc_demo.png")


if __name__ == "__main__":
    main()
