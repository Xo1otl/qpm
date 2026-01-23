import argparse

import jax

jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import super_gaussisan_opt as sg_opt
from bayes_opt import BayesianOptimization

from qpm import cwes2


def plot_bayesian_results(
    dk_scan: jax.Array,
    ref_int: jax.Array,
    opt_int: jax.Array,
    loss_hist: jax.Array,
    detected_range: tuple[float, float],
    plot_target: jax.Array,
    initial_int: jax.Array | None = None,
) -> None:
    """
    Generates and saves optimization result plots with DETECTED range overlay.
    """

    # Calculate target intensity from target amplitude profile for plotting
    target_profile_int = plot_target**2

    # Spectrum Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dk_scan, ref_int, "--", color="gray", alpha=0.5, label="Uniform Ref")
    # Plot detected range as a shaded region or span
    # We plot this BEHIND the main line
    if detected_range[1] > detected_range[0]:
        plt.axvspan(detected_range[0], detected_range[1], color="orange", alpha=0.2, label="Detected 95% Flat-Top")

    plt.plot(dk_scan, opt_int, "-", color="#2E86AB", linewidth=2, label="Optimized")

    if initial_int is not None:
        plt.plot(dk_scan, initial_int, "--", color="green", alpha=0.6, label="Initial 3-Seg")
    # Plot SG Target (just for reference of shape, not bounds)
    plt.plot(dk_scan, target_profile_int, "r:", linewidth=2, label="SG Target Shape")

    plt.legend()
    plt.title("Bayesian Optimization Result")
    plt.xlabel(r"$\Delta k$")
    plt.ylabel("Normalized Intensity")
    plt.savefig("bayesian_best_spectrum.png")
    plt.close()

    # Loss History Plot
    plt.figure(figsize=(10, 6))
    plt.plot(jnp.arange(len(loss_hist)), loss_hist, "-", color="#E63946")
    plt.yscale("log")
    plt.title("Optimization Loss History")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig("bayesian_best_loss.png")
    plt.close()


# Force CPU
def calculate_fw95m(x: jax.Array, y: jax.Array) -> tuple[float, tuple[float, float]]:
    """
    Calculate the full width at 95% of the maximum intensity.
    Returns the width of the WIDEST CONTINUOUS region above the threshold.
    This penalizes shapes with dips/ripples in the middle.
    """
    # Use numpy for efficient non-differentiable logic
    x_np = np.array(x)
    y_np = np.array(y)

    max_val = np.max(y_np)
    if max_val <= 1e-9:
        return 0.0

    threshold = 0.95 * max_val

    # Indices where intensity >= threshold
    valid_indices = np.where(y_np >= threshold)[0]

    if valid_indices.size == 0:
        return 0.0

    # Split into continuous regions (where index step > 1)
    # np.diff(valid_indices) gives differences between adjacent indices.
    # If difference > 1, it means there is a gap.
    split_locs = np.where(np.diff(valid_indices) > 1)[0] + 1
    regions = np.split(valid_indices, split_locs)

    max_continuous_width = 0.0

    for region in regions:
        if region.size > 0:
            # Width = Last - First in this region
            width = x_np[region[-1]] - x_np[region[0]]
            max_continuous_width = max(max_continuous_width, width)

    return float(max_continuous_width), (float(x_np[regions[0][0]]), float(x_np[regions[0][-1]])) if len(regions) > 0 and regions[0].size > 0 else (
        0.0,
        0.0,
    )


def run_bayesian_optimization(n_iter: int, init_points: int):
    # 1. Setup base config
    # We use ppln_opt default config as a base, then override specific parts
    # Or we can use sg_opt.get_default_config()
    base_cfg = sg_opt.get_default_config()

    # Pre-calculate fixed structures and reference intensity
    print("Pre-calculating reference structures...")
    initial_widths, kappa_vals = sg_opt.build_paper_structure(base_cfg)

    # Calculate Reference Intensity and Width
    dk_center = 2.0 * jnp.pi / base_cfg.grating_period
    # Use wider scan for reference calculation to capture full peaks
    # We define a consistent scan range for fair comparison
    scan_width_factor = 0.02  # 2% scan range
    dk_scan_range = jnp.linspace(dk_center * (1 - scan_width_factor / 2), dk_center * (1 + scan_width_factor / 2), 2000)
    dk_ref_scan = dk_scan_range

    # 1. Calculate Max Intensity from UNIFORM Grating of same length
    total_length = jnp.sum(initial_widths)
    num_periods = round(total_length / base_cfg.grating_period)
    half_p = base_cfg.grating_period / 2.0
    uniform_widths = jnp.tile(jnp.array([half_p, half_p]), num_periods)
    uniform_kappas = jnp.tile(jnp.array([base_cfg.kappa, -base_cfg.kappa]), num_periods)

    batch_sim = jax.vmap(cwes2.simulate_shg_npda, in_axes=(None, None, 0, None))

    # Simulate Uniform
    uniform_amps = batch_sim(uniform_widths, uniform_kappas, dk_ref_scan, base_cfg.b_initial[0])
    uniform_int_curve = jnp.abs(uniform_amps) ** 2
    max_intensity = float(jnp.max(uniform_int_curve))
    print(f"Reference Max Intensity (Uniform): {max_intensity:.4e}")

    # 2. Calculate Baseline Width from INITIAL 3-SEGMENT Grating
    # Simulate Initial
    initial_amps = batch_sim(jnp.abs(initial_widths), kappa_vals, dk_ref_scan, base_cfg.b_initial[0])
    initial_int_curve = jnp.abs(initial_amps) ** 2

    # Normalize by the UNIFORM max
    normalized_initial_curve = initial_int_curve / max_intensity

    # Calculate width
    ref_width_95, _ = calculate_fw95m(dk_ref_scan, normalized_initial_curve)
    initial_peak = float(jnp.max(normalized_initial_curve))
    print("Initial Structure Stats:")
    print(f"  - Peak Normalized Intensity: {initial_peak:.6f}")
    print(f"  - 95% Width (Baseline):      {ref_width_95:.6f}")

    if ref_width_95 <= 1e-9:
        print("Warning: Reference width is practically zero. Using 1.0 to avoid division by zero.")
        ref_width_95 = 1.0

    # 2. Define Objective Function
    def objective_function(target_width, target_center, target_intensity, sg_margin):
        # Calculate min/max from width and center
        half_width = target_width / 2.0
        target_min = target_center - half_width
        target_max = target_center + half_width

        target_flat_range = (target_min, target_max)

        # Update config with new parameters
        current_cfg = sg_opt.SimConfig(
            grating_period=base_cfg.grating_period,
            kappa=base_cfg.kappa,
            target_flat_range=target_flat_range,
            target_normalized_intensity=target_intensity,
            default_iterations=base_cfg.default_iterations,
            b_initial=base_cfg.b_initial,
            delta_k2=base_cfg.delta_k2,
        )

        # Fixed parameters
        sg_order = 6.0

        target_amp = jnp.sqrt(target_intensity)

        # Build the target profile for the inner optimizer
        opt_dk, opt_target = sg_opt.build_super_gaussian_target(
            current_cfg.target_flat_range, target_amp, points=60, margin=sg_margin, order=sg_order
        )

        # Run Inner Optimization (L-BFGS)
        # We start from the same initial simulation structure each time
        final_params, _, _ = sg_opt.run_optimization(
            current_cfg, initial_widths, kappa_vals, max_intensity, opt_dk, opt_target, iterations=current_cfg.default_iterations
        )

        # Simulate Resulting Spectrum for Evaluation
        final_widths_abs = jnp.abs(final_params)

        # Use the same WIDE scan range as reference to strictly prevent clipping
        # and ensure fair comparison of width
        dk_scan = dk_ref_scan

        batch_sim = jax.vmap(cwes2.simulate_shg_npda, in_axes=(None, None, 0, None))
        amps = batch_sim(final_widths_abs, kappa_vals, dk_scan, base_cfg.b_initial[0])
        intensity = jnp.abs(amps) ** 2

        # Calculate Metrics
        normalized_intensity_curve = intensity / max_intensity
        width_95, _ = calculate_fw95m(dk_scan, normalized_intensity_curve)

        # Objective: Maximize (95% bandwidth) relative to baseline
        # We use the actual peak of the result to ensure we don't pick zero-signal results.
        actual_peak = float(jnp.max(normalized_intensity_curve))

        score = 0.0 if actual_peak < 0.1 else width_95 / ref_width_95

        print(f"  [DEBUG] Peak: {actual_peak:.4f}, Width: {width_95:.6f}, Score: {score:.4f}")
        return float(score)

    # 3. Setup Bayesian Optimizer
    # Limits
    pbounds = {
        "target_width": (0.000934, 0.003),  # flat_range
        "target_center": (1.94425, 1.94625),  # center_range
        "target_intensity": (0.1, 0.3),  # normalized_intensity_range - Allow optimizer to find best trade-off
        "sg_margin": (0.2, 0.5),  # margin_range
    }

    optimizer = BayesianOptimization(f=objective_function, pbounds=pbounds, random_state=1, verbose=2)

    print("Starting Bayesian Optimization (Super-Gaussian)...")
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

    print("\nBest Result:")
    print(optimizer.max)

    # --- Re-run with Best Parameters to get Artifacts ---
    print("\nRe-running simulation with best parameters to generate plots...")
    best_params = optimizer.max["params"]

    # Extract params
    target_width = best_params["target_width"]
    target_center = best_params["target_center"]
    target_intensity = best_params["target_intensity"]
    sg_margin = best_params["sg_margin"]

    # Re-calculate derived
    half_width = target_width / 2.0
    target_min = target_center - half_width
    target_max = target_center + half_width
    target_flat_range = (target_min, target_max)

    # Config
    best_cfg = sg_opt.SimConfig(
        grating_period=base_cfg.grating_period,
        kappa=base_cfg.kappa,
        target_flat_range=target_flat_range,
        target_normalized_intensity=target_intensity,
        default_iterations=base_cfg.default_iterations,
        b_initial=base_cfg.b_initial,
        delta_k2=base_cfg.delta_k2,
    )

    target_amp = jnp.sqrt(target_intensity)
    sg_order = 6.0

    # 1. Build Optimization Target
    opt_dk, opt_target = sg_opt.build_super_gaussian_target(best_cfg.target_flat_range, target_amp, points=60, margin=sg_margin, order=sg_order)

    # 2. Run Optimization (L-BFGS)
    final_params, loss_hist, _ = sg_opt.run_optimization(
        best_cfg, initial_widths, kappa_vals, max_intensity, opt_dk, opt_target, iterations=best_cfg.default_iterations
    )

    # 3. Generate Plotting Data (Wide Scan)
    # Use a wider margin for the final plot to see context
    plot_dk, plot_target = sg_opt.build_super_gaussian_target(
        best_cfg.target_flat_range,
        target_amp,
        points=500,
        margin=1.0,
        order=8.0,  # Use higher order for visual "box" representation if desired, or same
    )

    # Simulate Best Result
    final_widths_abs = jnp.abs(final_params)
    batch_sim = jax.vmap(cwes2.simulate_shg_npda, in_axes=(None, None, 0, None))

    opt_amps = batch_sim(final_widths_abs, kappa_vals, plot_dk, base_cfg.b_initial[0])
    opt_int = (jnp.abs(opt_amps) ** 2) / max_intensity

    # Simulate Reference (Uniform)
    total_length = jnp.sum(initial_widths)
    num_periods = round(total_length / base_cfg.grating_period)
    half_p = base_cfg.grating_period / 2.0
    uniform_widths = jnp.tile(jnp.array([half_p, half_p]), num_periods)
    uniform_kappas = jnp.tile(jnp.array([base_cfg.kappa, -base_cfg.kappa]), num_periods)

    # RE-CALCULATE Normalization Reference to be absolutely safe
    # Ensure we capture the true peak of the uniform grating
    dk_norm_center = 2.0 * jnp.pi / base_cfg.grating_period
    # Wide scan for normalization
    dk_norm_scan = jnp.linspace(dk_norm_center * 0.98, dk_norm_center * 1.02, 5000)

    uniform_amps_norm = batch_sim(uniform_widths, uniform_kappas, dk_norm_scan, base_cfg.b_initial[0])
    max_intensity_final = float(jnp.max(jnp.abs(uniform_amps_norm) ** 2))
    print(f"Final Plotting Normalization (Max Intensity): {max_intensity_final:.4e}")

    # Now calculate curves for plotting using plot_dk
    ref_amps = batch_sim(uniform_widths, uniform_kappas, plot_dk, base_cfg.b_initial[0])
    ref_int = (jnp.abs(ref_amps) ** 2) / max_intensity_final

    # Simulate Initial 3-Segment (for plotting)
    initial_amps = batch_sim(jnp.abs(initial_widths), kappa_vals, plot_dk, base_cfg.b_initial[0])
    initial_int = (jnp.abs(initial_amps) ** 2) / max_intensity_final

    # Recalculate optimized intensity with correct norm
    opt_int = (jnp.abs(opt_amps) ** 2) / max_intensity_final

    # 4. Plot
    # Calculate metrics for the final plot
    normalized_intensity_curve = opt_int
    best_width_95, detected_range = calculate_fw95m(plot_dk, normalized_intensity_curve)
    best_peak = float(jnp.max(normalized_intensity_curve))

    print("-" * 40)
    print("Final Result Verification:")
    print(f"  Initial Width: {ref_width_95:.6f} (Peak: {float(jnp.max(initial_int)):.4f})")

    ratio = best_width_95 / ref_width_95
    print(f"  Best    Width: {best_width_95:.6f} (Ratio: {ratio:.2f})")
    print(f"  Best    Peak:  {best_peak:.6f}")

    if ratio < 1.0:
        print("\n  [WARNING] Optimization FAILED to improve upon the initial structure.")
        print("  The selected 'best' result is narrower than the baseline.")
        print("  This usually indicates the target constraints (e.g. Intensity) are physically impossible")
        print("  to achieve simultaneously with a broad bandwidth.")

    print(f"Detected 95% Flat-Top Range: {detected_range}")
    print("-" * 40)

    # We use our custom plot_bayesian_results
    plot_bayesian_results(plot_dk, ref_int, opt_int, loss_hist, detected_range, plot_target, initial_int)

    print("Saved best result plots to 'bayesian_best_spectrum.png' and 'bayesian_best_loss.png'")

    return optimizer.max


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=10, help="Number of BO iterations")
    parser.add_argument("--init", type=int, default=5, help="Number of BO initial points")
    args = parser.parse_args()

    run_bayesian_optimization(args.iter, args.init)
