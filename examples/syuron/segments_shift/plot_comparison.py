import argparse
import pickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from qpm import cwes2

# Force CPU to avoid VRAM contention if the optimizer is running
jax.config.update("jax_platforms", "cpu")

# =============================================================================
# Constants & Configuration
# =============================================================================
# Using values from ppln_opt.py / delta_k_distribution.py
GRATING_PERIOD = 3.23  # [µm]
KAPPA = 1.0e-5
TARGET_NORMALIZED_INTENSITY = 0.2
TARGET_FLAT_RANGE = (1.9440, 1.9465)

# 3-Segment Param defaults
N1 = 425
N2 = 2245
N3 = 425
DELTA1 = 1.28
DELTA2 = 1.95

B_INITIAL = jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex64)
DELTA_K2 = jnp.array(50.0)

# =============================================================================
# Helper Functions (Structure Construction)
# =============================================================================


def build_segment_domains(n_periods: int, period: float, kappa: float) -> tuple[jax.Array, jax.Array]:
    """Build domain widths and kappa values for a standard segment."""
    half_period = period / 2.0
    domain_widths = jnp.tile(jnp.array([half_period, half_period]), n_periods)
    kappa_vals = jnp.tile(jnp.array([kappa, -kappa]), n_periods)
    return domain_widths, kappa_vals


def build_phase_shift_domain(length: float, kappa: float) -> tuple[jax.Array, jax.Array]:
    """Build domain for a phase shift segment (fixed kappa = +1)."""
    return jnp.array([length]), jnp.array([kappa])


def build_3seg_structure() -> tuple[jax.Array, jax.Array]:
    """Build the complete 3-segment structure (Initial Guess)."""
    seg1_widths, seg1_kappa = build_segment_domains(N1, GRATING_PERIOD, KAPPA)
    ps1_widths, ps1_kappa = build_phase_shift_domain(DELTA1, KAPPA)
    seg2_widths, seg2_kappa = build_segment_domains(N2, GRATING_PERIOD, KAPPA)
    ps2_widths, ps2_kappa = build_phase_shift_domain(DELTA2, KAPPA)
    seg3_widths, seg3_kappa = build_segment_domains(N3, GRATING_PERIOD, KAPPA)

    domain_widths = jnp.concatenate([seg1_widths, ps1_widths, seg2_widths, ps2_widths, seg3_widths])
    kappa_vals = jnp.concatenate([seg1_kappa, ps1_kappa, seg2_kappa, ps2_kappa, seg3_kappa])
    return domain_widths, kappa_vals


def build_ppln_structure(total_length: float) -> tuple[jax.Array, jax.Array]:
    """Build a uniform PPLN structure with approximate total length."""
    num_periods = int(round(total_length / GRATING_PERIOD))
    return build_segment_domains(num_periods, GRATING_PERIOD, KAPPA)


def load_optimized_widths(filepath: str) -> jax.Array:
    """Load optimized params from pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded data keys: {data.keys()}")
    # Assuming standard structure from ppln_opt.py: {'params': ..., 'opt_state': ...}
    return data["params"]


# =============================================================================
# Simulation Logic
# =============================================================================


@jax.jit
def get_spectrum(widths, kappas, dk_scan):
    """
    Simulate TWM for a range of delta_k values.
    Returns intensity |A_sh|^2.
    """
    # Create batch simulation function
    # simulate_twm args: (domain_widths, n_vals, chi_vals, delta_k, delta_k2, b_input)
    # We map over delta_k (arg index 3)
    batch_sim = jax.vmap(cwes2.simulate_twm, in_axes=(None, None, None, 0, None, None))

    # Ensure widths are positive (optimization might explore negative, but physics is abs)
    real_widths = jnp.abs(widths)

    # Run simulation
    b_batch = batch_sim(real_widths, kappas, kappas, dk_scan, DELTA_K2, B_INITIAL)

    # Extract SHG (index 1), calculate intensity
    return jnp.abs(b_batch[:, 1]) ** 2


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Compare PPLN, 3-Segment, and Optimized Structures")
    parser.add_argument("pkl_file", nargs="?", default="45_60_02_v2.pkl", help="Path to optimized result pickle")
    args = parser.parse_args()

    print(f"Target file: {args.pkl_file}")

    # 1. Define Scan Range
    dk_center = 2.0 * jnp.pi / GRATING_PERIOD
    # Broad scan to see the lobes
    dk_scan = jnp.linspace(dk_center * 0.999, dk_center * 1.001, 1000)

    # 2. Construct Structures
    print("Constructing structures...")

    # A. 3-Segment (Initial)
    widths_3seg, kappas_structure = build_3seg_structure()
    len_3seg = jnp.sum(jnp.abs(widths_3seg))

    # B. Optimized
    # Note: Expects the pickle to contain 'params' which are the widths
    # The kappa structure (topology) remains the same as 3-segment
    try:
        widths_opt = load_optimized_widths(args.pkl_file)
        len_opt = jnp.sum(jnp.abs(widths_opt))
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    # C. PPLN (Reference)
    # Match length of the 3-segment/optimized structure roughly
    widths_ppln, kappas_ppln = build_ppln_structure(float(len_3seg))
    len_ppln = jnp.sum(jnp.abs(widths_ppln))

    print(f"\nStructure Lengths:")
    print(f"  PPLN:      {len_ppln:.4f} µm")
    print(f"  3-Segment: {len_3seg:.4f} µm")
    print(f"  Optimized: {len_opt:.4f} µm")

    # 3. Simulate
    print("\nRunning simulations...")

    # Reference PPLN Intensity for Normalization
    int_ppln = get_spectrum(widths_ppln, kappas_ppln, dk_scan)
    max_ppln = jnp.max(int_ppln)
    print(f"  Max PPLN Intensity: {max_ppln:.4e}")

    # 3-Segment Intensity
    int_3seg = get_spectrum(widths_3seg, kappas_structure, dk_scan)

    # Optimized Intensity
    int_opt = get_spectrum(widths_opt, kappas_structure, dk_scan)

    # 4. Plot
    print("\nPlotting...")
    plt.figure(figsize=(12, 7))

    # Normalize by PPLN max
    plt.plot(dk_scan, int_ppln / max_ppln, "k--", alpha=0.5, label=f"PPLN (Uniform) (L={len_ppln:.1f}µm)")
    plt.plot(dk_scan, int_3seg / max_ppln, "g-.", alpha=0.7, label=f"3-Segment (Init) (L={len_3seg:.1f}µm)")
    plt.plot(dk_scan, int_opt / max_ppln, "b-", linewidth=2, label=f"Optimized (L={len_opt:.1f}µm)")

    # Target Range Box
    plt.axvspan(TARGET_FLAT_RANGE[0], TARGET_FLAT_RANGE[1], color="orange", alpha=0.1, label="Target Range")
    plt.axhline(TARGET_NORMALIZED_INTENSITY, color="red", linestyle=":", label="Target Efficiency (0.2)")

    plt.xlabel(r"Mismatch $\Delta k$ [rad/µm]")
    plt.ylabel("Normalized Efficiency (relative to PPLN Max)")
    plt.title("Conversion Efficiency Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_file = "comparison_result.png"
    plt.savefig(out_file, dpi=300)
    print(f"Saved plot to {out_file}")


if __name__ == "__main__":
    main()
