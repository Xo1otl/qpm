import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from qpm import cwes2

# =============================================================================
# Geometric Constants from params.md
# =============================================================================
GRATING_PERIOD = 3.23  # [µm]
N1 = 425  # Segment 1 periods
N2 = 2245  # Segment 2 periods
N3 = 425  # Segment 3 periods
DELTA1 = 1.28  # Phase shift 1 length [µm]
DELTA2 = 1.95  # Phase shift 2 length [µm]

# Physics parameters
B_INITIAL = 1.0  # Fundamental amplitude
KAPPA_POSITIVE = 1.0e-5
KAPPA_NEGATIVE = -1.0e-5
TARGET_DK = 2.0 * jnp.pi / GRATING_PERIOD


def build_segment_domains(n_periods: int, period: float) -> tuple[jax.Array, jax.Array]:
    """Build domain widths and kappa values for a standard segment."""
    half_period = period / 2.0
    domain_widths = jnp.tile(jnp.array([half_period, half_period]), n_periods)
    kappa_vals = jnp.tile(jnp.array([KAPPA_POSITIVE, KAPPA_NEGATIVE]), n_periods)
    return domain_widths, kappa_vals


def build_phase_shift_domain(length: float) -> tuple[jax.Array, jax.Array]:
    """Build domain for a phase shift segment (fixed kappa = +1)."""
    return jnp.array([length]), jnp.array([KAPPA_POSITIVE])


def build_grating_structure() -> tuple[jax.Array, jax.Array]:
    """Build the complete 3-segment aperiodic QPM grating structure."""
    seg1_widths, seg1_kappa = build_segment_domains(N1, GRATING_PERIOD)
    ps1_widths, ps1_kappa = build_phase_shift_domain(DELTA1)
    seg2_widths, seg2_kappa = build_segment_domains(N2, GRATING_PERIOD)
    ps2_widths, ps2_kappa = build_phase_shift_domain(DELTA2)
    seg3_widths, seg3_kappa = build_segment_domains(N3, GRATING_PERIOD)

    domain_widths = jnp.concatenate([seg1_widths, ps1_widths, seg2_widths, ps2_widths, seg3_widths])
    kappa_vals = jnp.concatenate([seg1_kappa, ps1_kappa, seg2_kappa, ps2_kappa, seg3_kappa])
    return domain_widths, kappa_vals


def build_uniform_grating(total_length: float) -> tuple[jax.Array, jax.Array]:
    """Build a uniform periodic grating with the same total length."""
    n_periods = round(total_length / GRATING_PERIOD)
    return build_segment_domains(n_periods, GRATING_PERIOD)


def make_spectrum_evaluator(b_initial: jax.Array, delta_k2: jax.Array):
    def calculate_spectrum(widths, kappas, dk_values):
        def _simulate_single_dk(dk):
            # cwes2.simulate_twm returns [b_fw, b_sh, b_th], taking b_sh (index 1)
            return cwes2.simulate_twm(widths, kappas, kappas, dk, delta_k2, b_initial)[1]

        amplitude = jax.vmap(_simulate_single_dk)(dk_values)
        return jnp.abs(amplitude) ** 2

    return jax.jit(calculate_spectrum)


def main() -> None:
    # 1. Build Structures
    # Proposed Structure
    prop_widths, prop_kappas = build_grating_structure()
    total_length = jnp.sum(prop_widths)
    print(f"Structure Total Length: {total_length:.2f} µm")

    # Reference PPLN Structure
    ppln_widths, ppln_kappas = build_uniform_grating(float(total_length))
    print(f"PPLN Reference Length:  {jnp.sum(ppln_widths):.2f} µm")

    # 2. Simulation Setup
    b_initial = jnp.array([B_INITIAL, 0.0, 0.0], dtype=jnp.complex64)
    delta_k2 = jnp.array(50.0)  # Large mismatch for SFG suppression

    # Scan range centered at phase matching condition
    dk_center = TARGET_DK
    dk_scan = jnp.linspace(dk_center * 0.999, dk_center * 1.001, 1000)

    # 3. Calculate Spectra
    # Create the evaluator
    calculate_spectrum = make_spectrum_evaluator(b_initial, delta_k2)

    print("Simulating PPLN Reference...")
    ppln_intensity = calculate_spectrum(ppln_widths, ppln_kappas, dk_scan)
    max_ppln_intensity = jnp.max(ppln_intensity)
    print(f"Max PPLN Intensity: {max_ppln_intensity:.4e}")

    print("Simulating Proposed Structure...")
    prop_intensity = calculate_spectrum(prop_widths, prop_kappas, dk_scan)

    # 4. Normalize
    ppln_norm = ppln_intensity / max_ppln_intensity
    prop_norm = prop_intensity / max_ppln_intensity

    # 5. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dk_scan, ppln_norm, "--", label="Uniform PPLN", color="gray", alpha=0.7)
    plt.plot(dk_scan, prop_norm, "-", label="Proposed Structure", color="#2E86AB", linewidth=2)

    plt.xlabel(r"$\Delta k$ [1/µm]")
    plt.ylabel("Normalized Intensity (Relative to PPLN)")
    plt.title("Spectral Response: Proposed vs PPLN")
    plt.legend()
    plt.grid(visible=True, alpha=0.3)

    output_filename = "delta_k_distribution_comparison.png"
    plt.savefig(output_filename, dpi=150)
    print(f"Saved plot to: {output_filename}")


if __name__ == "__main__":
    main()
