import argparse
from collections.abc import Callable
from dataclasses import dataclass

import jax

# I don't have GPU
jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import super_gaussian_opt as sg_opt
from bayes_opt import BayesianOptimization

from qpm import cwes2

# --- Global Configuration ---
DK_CENTER = 1.94525
DK_HALF_WIDTH = 0.00175
DK_POINTS = 200  # Partition count
DK_START = DK_CENTER - DK_HALF_WIDTH
DK_END = DK_CENTER + DK_HALF_WIDTH

# Fixed global scan grid (Domain)
FIXED_DK_GRID = jnp.linspace(DK_START, DK_END, DK_POINTS)


# --- Factory for JIT Compilation ---
def make_jit_simulator(kappas: jax.Array, b_initial: jax.Array) -> Callable[[jax.Array, jax.Array], jax.Array]:
    def _sim_core(widths: jax.Array, dk_scan: jax.Array):
        return jax.vmap(cwes2.simulate_shg_npda, in_axes=(None, None, 0, None))(widths, kappas, dk_scan, b_initial)

    return jax.jit(_sim_core)


@dataclass
class OptimizationContext:
    """Holds constant data and the optimized simulator for the run."""

    base_cfg: sg_opt.SimConfig
    initial_widths: jax.Array
    kappa_vals: jax.Array
    max_intensity_ref: float
    b_fund: jax.Array
    simulator: Callable[[jax.Array, jax.Array], jax.Array]


@dataclass
class SimulationResult:
    """
    Encapsulates all results from a single simulation run.
    Eliminates the need for Union return types.
    """

    score: float
    normalized_intensity: jax.Array
    loss_history: jax.Array
    target_profile_visual: jax.Array  # For plotting purposes only


# --- Helper Functions ---
def calculate_fw95m(x: jax.Array, y: jax.Array) -> tuple[float, tuple[float, float]]:
    """It is rare for the regions exceeding 95% of the maximum within the FIXED_DK_GRID to be significantly fragmented."""
    max_val = jnp.max(y)
    if max_val <= 1e-9:
        return 0.0, (0.0, 0.0)

    threshold = 0.95 * max_val
    mask = y >= threshold
    dx = x[1] - x[0]
    total_width = jnp.sum(mask) * dx
    valid_indices = jnp.where(mask)[0]
    start_val = x[valid_indices[0]]
    end_val = x[valid_indices[-1]]

    return float(total_width), (float(start_val), float(end_val))


def get_reference_max_intensity(cfg: sg_opt.SimConfig, initial_widths: jax.Array) -> float:
    """Calculates the maximum intensity of a uniform reference grating."""
    total_length = jnp.sum(initial_widths)
    num_periods = round(total_length / cfg.grating_period)
    half_p = cfg.grating_period / 2.0

    uniform_widths = jnp.tile(jnp.array([half_p, half_p]), num_periods)
    uniform_kappas = jnp.tile(jnp.array([cfg.kappa, -cfg.kappa]), num_periods)

    ref_sim = make_jit_simulator(uniform_kappas, cfg.b_initial[0])
    uniform_amps = ref_sim(uniform_widths, FIXED_DK_GRID)
    return float(jnp.max(jnp.abs(uniform_amps) ** 2))


def plot_bayesian_results(
    ref_int: jax.Array,
    opt_int: jax.Array,
    loss_hist: jax.Array,
    detected_range: tuple[float, float],
    target_profile_int: jax.Array,
    initial_int: jax.Array | None = None,
) -> None:
    """Generates and saves optimization result plots."""

    # 1. Spectrum Plot
    plt.figure(figsize=(10, 6))
    plt.plot(FIXED_DK_GRID, ref_int, "--", color="gray", alpha=0.5, label="Uniform Ref")

    if detected_range[1] > detected_range[0]:
        plt.axvspan(detected_range[0], detected_range[1], color="orange", alpha=0.2, label="Detected 95% Flat-Top")

    plt.plot(FIXED_DK_GRID, opt_int, "-", color="#2E86AB", linewidth=2, label="Optimized")

    if initial_int is not None:
        plt.plot(FIXED_DK_GRID, initial_int, "--", color="green", alpha=0.6, label="Initial 3-Seg")

    # Plot the Rectangular Target
    plt.plot(FIXED_DK_GRID, target_profile_int, "r:", linewidth=2, label="Rectangular Target")

    plt.legend()
    plt.title("Bayesian Optimization Result (Flat-Top Target)")
    plt.xlabel(r"$\Delta k$")
    plt.ylabel("Normalized Intensity")
    plt.xlim(DK_START, DK_END)
    plt.savefig("bayesian_best_spectrum.png")
    plt.close()

    # 2. Loss History Plot
    plt.figure(figsize=(10, 6))
    plt.plot(jnp.arange(len(loss_hist)), loss_hist, "-", color="#E63946")
    plt.yscale("log")
    plt.title("Optimization Loss History")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss (Target Range Only)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig("bayesian_best_loss.png")
    plt.close()


# --- Main Optimization Logic ---
def run_full_simulation(
    target_width: float,
    target_intensity: float,
    ctx: OptimizationContext,
    ref_width_95: float,
) -> SimulationResult:
    """
    Runs the complete simulation and optimization pipeline.
    MSE is calculated ONLY within the target width.
    """
    target_amp = float(jnp.sqrt(target_intensity))
    half_width = target_width / 2.0

    range_start = DK_CENTER - half_width
    range_end = DK_CENTER + half_width

    current_cfg = sg_opt.SimConfig(
        grating_period=ctx.base_cfg.grating_period,
        kappa=ctx.base_cfg.kappa,
        target_flat_range=(range_start, range_end),
        target_normalized_intensity=target_intensity,
        default_iterations=ctx.base_cfg.default_iterations,
        b_initial=ctx.base_cfg.b_initial,
        delta_k2=ctx.base_cfg.delta_k2,
    )

    # --- ROI Filtering (Rectangular Window) ---
    # Create a mask strictly for the target range
    roi_mask = (range_start <= FIXED_DK_GRID) & (range_end >= FIXED_DK_GRID)
    dk_roi = FIXED_DK_GRID[roi_mask]
    target_roi = jnp.full_like(dk_roi, target_amp)
    # -------------------------------------------------------

    # Inner Optimization (L-BFGS)
    # The optimizer will only see the dk_roi points, effectively ignoring MSE outside this range.
    final_params, loss_hist, _ = sg_opt.run_optimization(
        current_cfg,
        ctx.initial_widths,
        ctx.kappa_vals,
        ctx.max_intensity_ref,
        dk_roi,
        target_roi,
        iterations=current_cfg.default_iterations,
    )

    # Evaluate
    final_widths_abs = jnp.abs(final_params)
    amps = ctx.simulator(final_widths_abs, FIXED_DK_GRID)
    normalized_intensity = (jnp.abs(amps) ** 2) / ctx.max_intensity_ref

    width_95, _ = calculate_fw95m(FIXED_DK_GRID, normalized_intensity)
    actual_peak = float(jnp.max(normalized_intensity))

    width_ratio = width_95 / ref_width_95
    score = width_ratio * actual_peak if actual_peak >= 0.05 else 0.0

    print(f"   -> FW95: {width_95:.6f} | Peak: {actual_peak:.4f} | Score: {score:.4f}")

    # Generate visual profile for plotting (0 outside, target_intensity inside)
    target_profile_visual = jnp.zeros_like(FIXED_DK_GRID)
    target_profile_visual = target_profile_visual.at[roi_mask].set(target_intensity)

    return SimulationResult(
        score=float(score), normalized_intensity=normalized_intensity, loss_history=loss_hist, target_profile_visual=target_profile_visual
    )


def run_bayesian_optimization(n_iter: int, init_points: int):
    # 1. Setup Phase
    base_cfg = sg_opt.get_default_config()
    base_cfg.default_iterations = 150

    print("Pre-calculating reference structures...")
    initial_widths, kappa_vals = sg_opt.build_paper_structure(base_cfg)
    max_intensity = get_reference_max_intensity(base_cfg, initial_widths)
    print(f"Reference Max Intensity (Uniform): {max_intensity:.4e}")

    print("Compiling JIT simulator kernel...")
    jit_simulator = make_jit_simulator(kappa_vals, base_cfg.b_initial[0])

    ctx = OptimizationContext(
        base_cfg=base_cfg,
        initial_widths=initial_widths,
        kappa_vals=kappa_vals,
        max_intensity_ref=max_intensity,
        b_fund=base_cfg.b_initial[0],
        simulator=jit_simulator,
    )

    # Baseline Stats
    initial_amps = ctx.simulator(jnp.abs(initial_widths), FIXED_DK_GRID)
    initial_norm = (jnp.abs(initial_amps) ** 2) / ctx.max_intensity_ref
    ref_width_95, _ = calculate_fw95m(FIXED_DK_GRID, initial_norm)

    if ref_width_95 <= 1e-9:
        ref_width_95 = 1.0

    # Calculate Initial Score
    initial_peak = float(jnp.max(initial_norm))
    initial_score = 1.0 * initial_peak if initial_peak >= 0.05 else 0.0

    print(f"Initial 95% Width (Baseline): {ref_width_95:.6f}")
    print(f"Initial Peak:                 {initial_peak:.6f}")
    print(f"Initial Score:                {initial_score:.6f}")

    # 2. Define Objective Wrapper
    def objective_wrapper(target_width: float, target_intensity: float) -> float:
        result = run_full_simulation(target_width, target_intensity, ctx, ref_width_95)
        return result.score

    # 3. Run Optimization
    pbounds = {
        "target_width": (0.0009, 0.003),
        "target_intensity": (0.15, 0.35),
    }

    optimizer = BayesianOptimization(f=objective_wrapper, pbounds=pbounds, random_state=1, verbose=2)

    # Inject the baseline parameters
    # Note: No conversion needed now, we probe the actual physical width we want.
    print("Probing with baseline parameters:")
    print(f"  Ref FW95 (Target Width): {ref_width_95:.6f}")

    optimizer.probe(
        params={"target_width": ref_width_95, "target_intensity": initial_peak},
        lazy=True,
    )

    print("Starting Bayesian Optimization...")
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    print("\nBest Result:")
    print(optimizer.max)

    # 4. Final Verification and Plotting
    best_params = optimizer.max["params"]  # pyright: ignore[reportOptionalSubscript]

    final_result = run_full_simulation(best_params["target_width"], best_params["target_intensity"], ctx, ref_width_95)

    # Reference spectrum for plotting
    num_periods = round(jnp.sum(ctx.initial_widths) / ctx.base_cfg.grating_period)
    half_p = ctx.base_cfg.grating_period / 2.0
    uniform_widths = jnp.tile(jnp.array([half_p, half_p]), num_periods)
    uniform_kappas = jnp.tile(jnp.array([ctx.base_cfg.kappa, -ctx.base_cfg.kappa]), num_periods)

    ref_sim = make_jit_simulator(uniform_kappas, ctx.b_fund)
    ref_amps = ref_sim(uniform_widths, FIXED_DK_GRID)
    ref_int = (jnp.abs(ref_amps) ** 2) / ctx.max_intensity_ref

    final_width, detected_range = calculate_fw95m(FIXED_DK_GRID, final_result.normalized_intensity)
    ratio = final_width / ref_width_95

    print("-" * 40)
    print(f"Final Width: {final_width:.6f} (Ratio: {ratio:.2f})")
    print(f"Final Peak:  {float(jnp.max(final_result.normalized_intensity)):.6f}")
    print(f"Final Score: {final_result.score:.6f}")
    print("-" * 40)

    plot_bayesian_results(
        ref_int,
        final_result.normalized_intensity,
        final_result.loss_history,
        detected_range,
        final_result.target_profile_visual,
        initial_norm,
    )
    print("Saved plots.")
    return optimizer.max


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=200, help="Number of BO iterations")
    parser.add_argument("--init", type=int, default=15, help="Number of BO initial points")
    args = parser.parse_args()

    run_bayesian_optimization(args.iter, args.init)
