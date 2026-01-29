import argparse
import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax

# Force CPU
jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import super_gaussian_opt as sg_opt
from qpm import cwes2

# --- Global Configuration (Must match BO script) ---
DK_CENTER = 1.94525
DK_HALF_WIDTH = 0.00175
DK_POINTS = 200
DK_START = DK_CENTER - DK_HALF_WIDTH
DK_END = DK_CENTER + DK_HALF_WIDTH

# Fixed global scan grid
FIXED_DK_GRID = jnp.linspace(DK_START, DK_END, DK_POINTS)


# --- JIT Factory ---
def make_jit_simulator(kappas: jax.Array, b_initial: jax.Array) -> Callable[[jax.Array, jax.Array], jax.Array]:
    def _sim_core(widths: jax.Array, dk_scan: jax.Array):
        return jax.vmap(cwes2.simulate_shg_npda, in_axes=(None, None, 0, None))(widths, kappas, dk_scan, b_initial)

    return jax.jit(_sim_core)


# --- Helper Functions ---
def calculate_fw95m(x: jax.Array, y: jax.Array) -> tuple[float, tuple[float, float]]:
    max_val = jnp.max(y)
    if max_val <= 1e-9:
        return 0.0, (0.0, 0.0)

    threshold = 0.95 * max_val
    mask = y >= threshold
    dx = x[1] - x[0]
    total_width = jnp.sum(mask) * dx

    valid_indices = jnp.where(mask)[0]
    if len(valid_indices) == 0:
        return 0.0, (0.0, 0.0)

    start_val = x[valid_indices[0]]
    end_val = x[valid_indices[-1]]

    return float(total_width), (float(start_val), float(end_val))


def get_reference_max_intensity(cfg: sg_opt.SimConfig, initial_widths: jax.Array) -> tuple[float, jax.Array]:
    """Calculates max intensity of uniform reference and returns its spectrum."""
    total_length = jnp.sum(initial_widths)
    num_periods = round(total_length / cfg.grating_period)
    half_p = cfg.grating_period / 2.0

    uniform_widths = jnp.tile(jnp.array([half_p, half_p]), num_periods)
    uniform_kappas = jnp.tile(jnp.array([cfg.kappa, -cfg.kappa]), num_periods)

    ref_sim = make_jit_simulator(uniform_kappas, cfg.b_initial[0])
    uniform_amps = ref_sim(uniform_widths, FIXED_DK_GRID)
    return float(jnp.max(jnp.abs(uniform_amps) ** 2)), uniform_amps


def analyze_trace(x: jax.Array, y: jax.Array, widths: jax.Array | None = None) -> dict[str, Any]:
    """Helper to extract metrics for plotting."""
    fw95, (start, end) = calculate_fw95m(x, y)
    peak = float(jnp.max(y))
    length = float(jnp.sum(widths)) if widths is not None else 0.0
    return {
        "fw95": fw95,
        "peak": peak,
        "length": length,
        "region_start": start,
        "region_end": end,
    }


def plot_verification_results(data: dict[str, Any], output_filename: str = "verify_result.png", scale: float = 1.0) -> None:
    """Generates the comparison plot from data dictionary."""
    grid = FIXED_DK_GRID
    fs = 12 * scale

    # Extract Data
    ref_norm = jnp.sqrt(data["spectrum_ref"])
    init_norm = jnp.sqrt(data["spectrum_initial"])
    final_norm = jnp.sqrt(data["spectrum_final"])

    init_widths = data["initial_widths"]
    final_widths = data["final_widths"]

    # Calculate Metrics
    m_ref = analyze_trace(grid, ref_norm)
    m_init = analyze_trace(grid, init_norm, init_widths)
    m_final = analyze_trace(grid, final_norm, final_widths)

    # Keep figure size constant to ensure scale affects relative text size
    plt.figure(figsize=(12, 7))

    # Trace 1: Uniform
    label_ref = f"Uniform (W={m_ref['fw95']:.5f}, A={m_ref['peak']:.2f})"
    plt.plot(grid, ref_norm, ":", color="gray", alpha=0.5, label=label_ref)

    # Trace 2: Initial 3-Seg
    label_init = f"Initial 3-Seg (L={m_init['length']:.1f}, W={m_init['fw95']:.5f}, A={m_init['peak']:.2f})"
    plt.plot(grid, init_norm, "--", color="green", alpha=0.6, label=label_init)

    # Trace 3: Optimized
    label_opt = f"Optimized (L={m_final['length']:.1f}, W={m_final['fw95']:.5f}, A={m_final['peak']:.2f})"
    plt.plot(grid, final_norm, "-", color="#2E86AB", linewidth=2 * scale, label=label_opt)

    plt.legend(fontsize=fs * 0.9)
    plt.title("Optimization Verification", fontsize=fs * 1.2)
    plt.xlabel(r"$\Delta k$", fontsize=fs)
    plt.ylabel("Normalized Amplitude", fontsize=fs)
    plt.xticks(fontsize=fs * 0.8)
    plt.yticks(fontsize=fs * 0.8)
    plt.xlim(DK_START, DK_END)
    plt.grid(visible=True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Plot saved to {output_filename}")
    plt.close()


def plot_domain_widths(widths: jax.Array, output_filename: str = "domain_widths.png", scale: float = 1.0) -> None:
    """Generates a scatter plot of domain widths to avoid visual saturation."""
    widths_np = np.array(widths)
    indices = np.arange(len(widths_np))
    fs = 12 * scale

    # Keep figure size constant to ensure scale affects relative text size
    plt.figure(figsize=(10, 6))
    # Using scatter instead of bar to prevent "solid color" block with many domains
    plt.scatter(indices, widths_np, color="#2E86AB", s=0.5 * scale, alpha=0.7)

    plt.title("Optimized Domain Width Distribution", fontsize=fs * 1.2)
    plt.xlabel("Domain Index", fontsize=fs)
    plt.ylabel(r"Width ($\mu m$)", fontsize=fs)
    plt.xticks(fontsize=fs * 0.8)
    plt.yticks(fontsize=fs * 0.8)
    plt.grid(visible=True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Domain width plot saved to {output_filename}")
    plt.close()


# --- Main Verification Logic ---
def run_verification(target_width: float, target_intensity: float, iterations: int, scale: float = 1.0) -> None:
    """Executes the optimization and saves the results."""
    print("--- Starting Verification ---")
    print(f"Target Width:      {target_width:.6f}")
    print(f"Target Intensity: {target_intensity:.6f}")
    print(f"L-BFGS Iterations: {iterations}")

    # 1. Setup Base Config & Structures
    base_cfg = sg_opt.get_default_config()
    base_cfg.default_iterations = iterations

    print("Building initial 3-segment structure...")
    initial_widths, kappa_vals = sg_opt.build_paper_structure(base_cfg)

    # 2. Reference Calculations (Periodic Polling)
    max_intensity_ref, ref_amps = get_reference_max_intensity(base_cfg, initial_widths)
    ref_norm_spectrum = (jnp.abs(ref_amps) ** 2) / max_intensity_ref
    print(f"Reference Max Intensity (Uniform): {max_intensity_ref:.4e}")

    # 3. Simulator & Baseline Stats
    jit_simulator = make_jit_simulator(kappa_vals, base_cfg.b_initial[0])

    # Initial 3-segment simulation
    initial_amps = jit_simulator(jnp.abs(initial_widths), FIXED_DK_GRID)
    initial_norm = (jnp.abs(initial_amps) ** 2) / max_intensity_ref

    initial_fw95, _ = calculate_fw95m(FIXED_DK_GRID, initial_norm)
    # Handle edge case where initial is too narrow/low
    baseline_fw95_ref = initial_fw95 if initial_fw95 > 1e-9 else 1.0

    print(f"Baseline (3-Seg) FW95: {initial_fw95:.6f}")

    # 4. Prepare Optimization Target
    target_amp = float(jnp.sqrt(target_intensity))
    half_width = target_width / 2.0
    range_start = DK_CENTER - half_width
    range_end = DK_CENTER + half_width

    # ROI Masking
    roi_mask = (range_start <= FIXED_DK_GRID) & (range_end >= FIXED_DK_GRID)
    dk_roi = FIXED_DK_GRID[roi_mask]
    target_roi = jnp.full_like(dk_roi, target_amp)

    current_cfg = sg_opt.SimConfig(
        grating_period=base_cfg.grating_period,
        kappa=base_cfg.kappa,
        target_flat_range=(range_start, range_end),
        target_normalized_intensity=target_intensity,
        default_iterations=iterations,
        b_initial=base_cfg.b_initial,
        delta_k2=base_cfg.delta_k2,
    )

    # 5. Run L-BFGS
    print("Running L-BFGS optimization...")
    final_params, loss_hist, _ = sg_opt.run_optimization(
        current_cfg,
        initial_widths,
        kappa_vals,
        max_intensity_ref,
        dk_roi,
        target_roi,
        iterations=iterations,
    )

    # 6. Final Evaluation
    final_widths_abs = jnp.abs(final_params)
    final_amps = jit_simulator(final_widths_abs, FIXED_DK_GRID)
    final_norm = (jnp.abs(final_amps) ** 2) / max_intensity_ref

    final_fw95, (_, _) = calculate_fw95m(FIXED_DK_GRID, final_norm)
    final_peak = float(jnp.max(final_norm))

    # Score calculation matching BO script
    width_ratio = final_fw95 / baseline_fw95_ref
    score = width_ratio * final_peak if final_peak >= 0.05 else 0.0

    print("-" * 40)
    print(f"Final FW95: {final_fw95:.6f} (Ratio: {width_ratio:.2f})")
    print(f"Final Peak: {final_peak:.6f}")
    print(f"Final Score: {score:.6f}")
    print("-" * 40)

    # 7. Save Data
    data = {
        "params": {"target_width": target_width, "target_intensity": target_intensity, "iterations": iterations},
        "initial_widths": initial_widths,
        "final_widths": final_widths_abs,
        "loss_history": loss_hist,
        "spectrum_final": final_norm,
        "spectrum_initial": initial_norm,
        "spectrum_ref": ref_norm_spectrum,
        "metrics": {"fw95": final_fw95, "peak": final_peak, "score": score},
    }

    pkl_filename = "verify_data.pkl"
    with open(pkl_filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Data saved to {pkl_filename}")

    # 8. Plot
    plot_verification_results(data, scale=scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify L-BFGS optimization.")

    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Run a fresh optimization")
    group.add_argument("--load", type=str, help="Load and plot an existing pickle file")

    # Shared args
    parser.add_argument("--scale", type=float, default=1.2, help="Scale factor for plot font sizes (default: 1.2)")

    # Args for --run mode
    parser.add_argument("--width", type=float, help="Target spectral width (required for --run)")
    parser.add_argument("--int", type=float, help="Target normalized intensity (required for --run)")
    parser.add_argument("--iter", type=int, default=500, help="L-BFGS iterations")

    args = parser.parse_args()

    if args.load:
        load_path = Path(args.load)
        if not load_path.exists():
            print(f"Error: File {args.load} not found.")
            sys.exit(1)

        print(f"Loading data from {args.load}...")
        with load_path.open("rb") as f:
            data = pickle.load(f)

        # Generate plot names
        base_name = load_path.stem
        plot_name = f"{base_name}_plot.png"
        plot_verification_results(data, output_filename=plot_name, scale=args.scale)

        width_plot_name = f"{base_name}_widths.png"
        plot_domain_widths(data["final_widths"], output_filename=width_plot_name, scale=args.scale)

    elif args.run:
        if args.width is None or args.int is None:
            parser.error("--run requires --width and --int")

        run_verification(args.width, args.int, args.iter, scale=args.scale)
