import argparse
import os
import pickle
from dataclasses import dataclass

import jax

jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax_tqdm import scan_tqdm  # pyright: ignore[reportPrivateImportUsage]

from qpm import cwes2


MARGIN = 0.2316097145034998


@dataclass
class SimConfig:
    """Configuration for PPLN simulation and optimization."""

    grating_period: float
    kappa: float
    target_flat_range: tuple[float, float]
    target_normalized_intensity: float
    default_iterations: int
    b_initial: jax.Array
    delta_k2: jax.Array


def get_default_config() -> SimConfig:
    width = 0.0018650943254284217
    center = 1.9458495445964978
    intensity = 0.20536506134153698
    flat_min = center - width / 2
    flat_max = center + width / 2
    return SimConfig(
        grating_period=3.23,
        kappa=1.0e-5,
        target_flat_range=(flat_min, flat_max),
        target_normalized_intensity=intensity,
        default_iterations=300,
        b_initial=jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex64),
        delta_k2=jnp.array(50.0),
    )


# --- Helper Functions: Structure Building ---
def build_segment_domains(n_periods: int, period: float, kappa: float) -> tuple[jax.Array, jax.Array]:
    half_period = period / 2.0
    domain_widths = jnp.tile(jnp.array([half_period, half_period]), n_periods)
    kappa_vals = jnp.tile(jnp.array([kappa, -kappa]), n_periods)
    return domain_widths, kappa_vals


def build_phase_shift_domain(length: float, kappa: float) -> tuple[jax.Array, jax.Array]:
    return jnp.array([length]), jnp.array([kappa])


def build_paper_structure(cfg: SimConfig) -> tuple[jax.Array, jax.Array]:
    n1 = 425
    n2 = 2245
    n3 = 425
    delta1 = 1.28
    delta2 = 1.95

    seg1_widths, seg1_kappa = build_segment_domains(n1, cfg.grating_period, cfg.kappa)
    ps1_widths, ps1_kappa = build_phase_shift_domain(delta1, cfg.kappa)
    seg2_widths, seg2_kappa = build_segment_domains(n2, cfg.grating_period, cfg.kappa)
    ps2_widths, ps2_kappa = build_phase_shift_domain(delta2, cfg.kappa)
    seg3_widths, seg3_kappa = build_segment_domains(n3, cfg.grating_period, cfg.kappa)

    domain_widths = jnp.concatenate([seg1_widths, ps1_widths, seg2_widths, ps2_widths, seg3_widths])
    kappa_vals = jnp.concatenate([seg1_kappa, ps1_kappa, seg2_kappa, ps2_kappa, seg3_kappa])
    return domain_widths, kappa_vals


# --- Helper Functions: Checkpointing ---
def save_checkpoint(filename, params, opt_state):
    data = {"params": params, "opt_state": opt_state}
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"State saved to '{filename}'")


def load_checkpoint(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"State loaded from '{filename}'")
    return data["params"], data["opt_state"]


# --- Core Logic: Target Generation & Optimization ---


def build_super_gaussian_target(
    target_range: tuple[float, float], target_amp: float, points: int = 100, margin: float = 0.5, order: float = 6.0
) -> tuple[jax.Array, jax.Array]:
    """
    Generates a Super-Gaussian spectral target profile.

    Args:
        target_range: (min_dk, max_dk) where response should be flat.
        target_amp: Peak normalized amplitude value (e.g. sqrt(0.26)).
        points: Number of discrete dk points in the scan.
        margin: Fractional margin to add to scan range (0.5 = 50% wider on each side).
        order: Super-Gaussian order (higher = squarer).

    Returns:
        (dk_scan, target_profile_amp)
    """
    center = sum(target_range) / 2.0
    width = target_range[1] - target_range[0]

    # Define scan grid with margins
    scan_span = width * (1 + 2 * margin)
    dk_scan = jnp.linspace(center - scan_span / 2, center + scan_span / 2, points)

    # Calculate sigma such that amplitude is ~95% at the edges of target_range
    # Formula derived from A = exp(-(|x|/sigma)^N)
    edge_val = 0.95
    half_width = width / 2.0
    sigma = half_width / ((-jnp.log(edge_val)) ** (1 / order))

    # Generate Profile (Amplitude)
    profile = target_amp * jnp.exp(-(jnp.abs((dk_scan - center) / sigma) ** order))

    return dk_scan, profile


def make_optimization_step(
    optimizer: optax.GradientTransformation,
    dk_targets: jax.Array,
    target_profile: jax.Array,
    kappa_vals: jax.Array,
    b_initial: jax.Array,
    max_intensity_ref: float,
    iterations: int,
):
    """Creates the JIT-compiled optimization scan loop using Shape-based MSE."""
    max_amp_ref = jnp.sqrt(max_intensity_ref)
    b_fund = b_initial[0]

    # Vectorized simulation over delta_k (arg index 2)
    batch_sim = jax.vmap(cwes2.simulate_shg_npda, in_axes=(None, None, 0, None))

    def loss_fn(params_widths):
        # Enforce physical constraints (widths > 0)
        real_widths = jnp.abs(params_widths)

        # 1. Batched Simulation
        shg_amps_complex = batch_sim(real_widths, kappa_vals, dk_targets, b_fund)

        # 2. Normalize Magnitude
        current_amp = jnp.abs(shg_amps_complex) / max_amp_ref

        # 3. Shape Error (MSE against Super-Gaussian vector)
        return jnp.mean((current_amp - target_profile) ** 2)

    @jax.jit
    @scan_tqdm(iterations, print_rate=10)
    def step_scan(state, i):
        params, opt_state = state
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params, value=loss_val, grad=grads, value_fn=loss_fn)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss_val

    return step_scan


def get_reference_intensity(cfg: SimConfig, initial_widths: jax.Array) -> float:
    """Calculates the maximum intensity of a uniform reference grating."""
    total_length = jnp.sum(initial_widths)
    num_uniform_periods = int(round(total_length / cfg.grating_period))

    half_p = cfg.grating_period / 2.0
    uniform_widths = jnp.tile(jnp.array([half_p, half_p]), num_uniform_periods)
    uniform_kappas = jnp.tile(jnp.array([cfg.kappa, -cfg.kappa]), num_uniform_periods)

    dk_center = 2.0 * jnp.pi / cfg.grating_period
    dk_ref_scan = jnp.linspace(dk_center * 0.99, dk_center * 1.01, 500)

    batch_sim = jax.vmap(cwes2.simulate_shg_npda, in_axes=(None, None, 0, None))
    b_fund = cfg.b_initial[0]

    ref_amps = batch_sim(uniform_widths, uniform_kappas, dk_ref_scan, b_fund)
    max_intensity = float(jnp.max(jnp.abs(ref_amps) ** 2))
    return max_intensity


def run_optimization(
    cfg: SimConfig,
    initial_params: jax.Array,
    kappa_vals: jax.Array,
    max_intensity: float,
    dk_scan: jax.Array,
    target_profile: jax.Array,
    iterations: int,
    resume_path: str = None,
) -> tuple[jax.Array, jax.Array, optax.OptState]:
    """Runs the optimization loop with injected target profile."""

    print(f"\nSetting up L-BFGS (Iterations: {iterations})...")
    optimizer = optax.lbfgs(learning_rate=1.0)

    # --- State Initialization (New or Resume) ---
    current_params = initial_params
    current_opt_state = optimizer.init(current_params)

    if resume_path and os.path.exists(resume_path):
        current_params, current_opt_state = load_checkpoint(resume_path)
    elif resume_path:
        print(f"Warning: Checkpoint '{resume_path}' not found. Starting fresh.")

    # --- Optimization Factory ---
    step_fn = make_optimization_step(
        optimizer,
        dk_scan,
        target_profile,
        kappa_vals,
        cfg.b_initial,
        max_intensity,
        iterations,
    )

    print("Starting optimization scan...")
    (final_params, final_opt_state), loss_hist = jax.lax.scan(step_fn, (current_params, current_opt_state), jnp.arange(iterations))

    return final_params, loss_hist, final_opt_state


def plot_results(
    dk_scan: jax.Array,
    ref_int: jax.Array,
    opt_int: jax.Array,
    loss_hist: jax.Array,
    target_range: tuple[float, float],
    target_profile_amp: jax.Array,
) -> None:
    """Generates and saves optimization result plots with target overlay."""

    # Calculate target intensity from target amplitude profile for plotting
    target_profile_int = target_profile_amp**2

    # Spectrum Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dk_scan, ref_int, "--", color="gray", alpha=0.5, label="Uniform Ref")
    plt.plot(dk_scan, opt_int, "-", color="#2E86AB", linewidth=2, label="Optimized")

    # Plot the exact Super-Gaussian target used by the optimizer
    # (Note: dk_scan passed here is the wider plot scan, so we must be careful.
    # Ideally, we plot the target profile over the scan range used for optimization,
    # but here we just overlay the target range box for visual simplicity
    # or plotting the SG if dimensions match.)

    # Simple visual guides
    plt.axvspan(*target_range, color="g", alpha=0.1, label="Target Range")
    plt.plot(dk_scan, target_profile_int, "r:", linewidth=2, label="SG Target")

    plt.legend()
    plt.title("PPLN Optimization Result (Super-Gaussian Target)")
    plt.xlabel(r"$\Delta k$")
    plt.ylabel("Normalized Intensity")
    plt.savefig("ppln_optimized.png")
    plt.close()

    # Loss History Plot
    plt.figure(figsize=(10, 6))
    plt.plot(jnp.arange(len(loss_hist)), loss_hist, "-", color="#E63946")
    plt.yscale("log")
    plt.title("Optimization Loss History (Shape MSE)")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig("loss_history.png")
    plt.close()


def main():
    # --- Config ---
    cfg = get_default_config()

    # --- CLI Arguments ---
    parser = argparse.ArgumentParser(description="PPLN Optimization with Super-Gaussian Target")
    parser.add_argument("--resume", type=str, default=None, help="Path to .pkl to resume.")
    parser.add_argument("--save", type=str, default="checkpoint.pkl", help="Path to save result.")
    parser.add_argument("--iters", type=int, default=cfg.default_iterations, help="Iterations.")
    args = parser.parse_args()

    # --- Setup ---
    print("Building 3-segment initial structure...")
    initial_widths, kappa_vals = build_paper_structure(cfg)

    # --- Normalization ---
    print("Calculating reference maximum intensity...")
    max_intensity = get_reference_intensity(cfg, initial_widths)
    print(f"Max Intensity (Reference): {max_intensity:.4e}")

    # --- Build Target Profile ---
    print("Building Super-Gaussian target...")
    target_amp = jnp.sqrt(cfg.target_normalized_intensity)

    # We create two sets of targets:
    # 1. opt_dk, opt_target: The dense, narrow grid used for the optimizer
    # 2. plot_dk, plot_target: A wider grid for final visualization

    # Optimization Grid (Narrower, focused on constraints)
    opt_dk, opt_target = build_super_gaussian_target(cfg.target_flat_range, target_amp, points=60, margin=MARGIN, order=6.0)

    # --- Run Optimization ---
    final_params, loss_hist, final_opt_state = run_optimization(
        cfg,
        initial_widths,
        kappa_vals,
        max_intensity,
        opt_dk,
        opt_target,
        args.iters,
        args.resume,
    )

    final_params_abs = jnp.abs(final_params)
    print(f"Final Loss: {loss_hist[-1]:.6e}")

    # --- Save Checkpoint ---
    save_checkpoint(args.save, final_params, final_opt_state)

    # --- Final Simulation for Plotting ---
    print("Generating plots...")

    # Create a wider scan for the final plot to see the floor
    plot_dk, plot_target = build_super_gaussian_target(
        cfg.target_flat_range,
        target_amp,
        points=500,
        margin=1.0,  # Wider view
        order=8.0,
    )

    batch_sim = jax.vmap(cwes2.simulate_shg_npda, in_axes=(None, None, 0, None))
    b_fund = cfg.b_initial[0]

    # Reference Spectrum
    total_length = jnp.sum(initial_widths)
    num_uniform_periods = round(total_length / cfg.grating_period)
    half_p = cfg.grating_period / 2.0
    uniform_widths = jnp.tile(jnp.array([half_p, half_p]), num_uniform_periods)
    uniform_kappas = jnp.tile(jnp.array([cfg.kappa, -cfg.kappa]), num_uniform_periods)

    ref_amps = batch_sim(uniform_widths, uniform_kappas, plot_dk, b_fund)
    ref_int = (jnp.abs(ref_amps) ** 2) / max_intensity

    # Optimized Spectrum
    opt_amps = batch_sim(final_params_abs, kappa_vals, plot_dk, b_fund)
    opt_int = (jnp.abs(opt_amps) ** 2) / max_intensity

    plot_results(plot_dk, ref_int, opt_int, loss_hist, cfg.target_flat_range, plot_target)
    print("Done.")


if __name__ == "__main__":
    main()
