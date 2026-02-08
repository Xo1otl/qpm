import argparse

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp  # pyright: ignore[reportUnknownVariableType]

from qpm import mgoslt

# Enable x64 (consistent with other scripts)
jax.config.update("jax_enable_x64", True)


def setup_font_scaling(scalar: float) -> None:
    """Scales all matplotlib font sizes by a scalar."""
    base_size = 12
    scaled_size = base_size * scalar

    plt.rc("font", size=scaled_size)
    plt.rc("axes", titlesize=scaled_size * 1.2)
    plt.rc("axes", labelsize=scaled_size)
    plt.rc("xtick", labelsize=scaled_size * 3)
    plt.rc("ytick", labelsize=scaled_size * 3)
    plt.rc("legend", fontsize=scaled_size)
    plt.rc("figure", titlesize=scaled_size * 1.5)


def get_parameters() -> tuple[float, float, float]:
    """Returns (delta_k, coherence_length, kappa)."""
    wl = 1.064
    T = 70.0
    dk_val = float(mgoslt.calc_twm_delta_k(wl, wl, T))
    lc = np.pi / dk_val
    # kappa value from plot_results_for_slide.py
    # kappa_shg_val = 1.5e-5 / (2 / jnp.pi) -> but that was for effective d_eff in Fourier expansion context?
    # actually, standard coupled equations use d_eff directly or kappa.
    # In plot_results_for_slide.py: kappa_shg_val = 1.5e-5 / (2 / jnp.pi)
    # This looks like it scales the effective nonlinearity back to the physical one because 2/pi is the first Fourier coefficient of a square wave.
    # So the physical kappa within a domain is likely 1.5e-5 / (2/pi) = 1.5e-5 * pi / 2.
    # Let's use that value to be consistent with the other script's "kappa_shg_val" which was likely used as the effective kappa for the "avg" model,
    # but here we are doing domain-by-domain, so we need the local kappa.
    # Wait, in plot_results_for_slide.py:
    # signs = np.tile([1.0, -1.0], ...)
    # k1 = signs * kappa_shg_val
    # The simulation `simulate_magnus_with_trace` uses these local kappa values.
    # So `kappa_shg_val` in that script IS the local kappa magnitude.
    #
    # Let's re-read plot_results_for_slide.py carefully.
    # kappa_shg_val = 1.5e-5 / (2 / jnp.pi)
    # k1 = jnp.array(signs * kappa_shg_val)
    # So the local kappa is indeed 1.5e-5 / (2/pi).

    kappa_val = 1.5e-5 / (2.0 / np.pi)

    return dk_val, lc, kappa_val


def shg_ode(z: float, A: np.ndarray, kappa: float, dk: float) -> np.ndarray:
    """
    Coupled wave equations for SHG.
    A = [A1, A2] (Fundamental, Second Harmonic)
    dA1/dz = i * kappa * A1* * A2 * exp(-i * dk * z)
    dA2/dz = i * kappa * A1^2 * exp(i * dk * z)
    """
    A1 = A[0]
    A2 = A[1]

    # We need to handle complex numbers. scipy.integrate handles complex IVPs if initial value is complex.
    dA1dz = 1j * kappa * np.conj(A1) * A2 * np.exp(-1j * dk * z)
    dA2dz = 1j * kappa * A1**2 * np.exp(1j * dk * z)

    return np.array([dA1dz, dA2dz])


def run_simulation(enable_qpm: bool = True) -> tuple[np.ndarray, np.ndarray]:
    dk, lc, kappa_mag = get_parameters()

    # Initial conditions
    # A1(0) = sqrt(10) (Power ~ 10 a.u., magnitude)
    # A2(0) = 0
    y0 = np.array([np.sqrt(10.0) + 0j, 0.0 + 0j])

    z_all = [0.0]
    a2_all = [0.0]

    current_z = 0.0
    current_y = y0

    # 5 periods = 10 domains
    num_periods = 5
    num_domains = num_periods * 2

    for i in range(num_domains):
        # Determine sign of kappa (starts with +1 usually for QPM)
        if enable_qpm:
            sign = 1.0 if i % 2 == 0 else -1.0
        else:
            sign = 1.0

        kappa_local = sign * kappa_mag

        z_start = current_z
        z_end = current_z + lc

        # Solve for this domain
        # Using RK45
        res = solve_ivp(
            fun=lambda z, y: shg_ode(z, y, kappa_local, dk),
            t_span=(z_start, z_end),
            y0=current_y,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            max_step=lc / 50.0,  # Ensure enough resolution
        )

        # Append results (excluding the very first point to avoid duplicate if stiching,
        # but solve_ivp includes start point. Let's slice off start point except for very first domain)
        if i == 0:
            z_all = list(res.t)
            a2_all = list(np.abs(res.y[1]))
        else:
            z_all.extend(res.t[1:])
            a2_all.extend(np.abs(res.y[1])[1:])

        current_z = z_end
        current_y = res.y[:, -1]

    return np.array(z_all), np.array(a2_all)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot microscopic SHG evolution.")
    parser.add_argument("--scalar", type=float, default=1.0, help="Font size scaling factor.")
    parser.add_argument("--output", type=str, default="micro_shg.png", help="Output filename.")
    args = parser.parse_args()

    setup_font_scaling(args.scalar)

    print("Running simulation (QPM)...")
    z, sh_amp = run_simulation(enable_qpm=True)

    print("Running simulation (No QPM)...")
    z_no_qpm, sh_amp_no_qpm = run_simulation(enable_qpm=False)

    print(f"Simulation complete. Z max: {z.max():.2f}, SH max: {sh_amp.max():.4f}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # Plot SH amplitude in green
    ax.plot(z, sh_amp, color="green", linewidth=2.0)  # QPM
    ax.plot(z_no_qpm, sh_amp_no_qpm, color="green", linewidth=2.0, linestyle="--")  # No QPM
    # "SH振幅線の色は緑" -> "Color of SH amplitude line is green"

    ax.set_xlabel("Position (µm)")
    ax.set_ylabel("SH Amplitude")

    # Grid
    ax.grid(True, alpha=0.3)

    # Save
    fig.savefig(args.output, dpi=300)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
