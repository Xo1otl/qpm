import argparse
import os
import pickle

import jax

jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax_tqdm import scan_tqdm  # pyright: ignore[reportPrivateImportUsage]

from dataclasses import dataclass

from qpm import cwes2


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
    """Returns the default simulation configuration."""
    return SimConfig(
        grating_period=3.23,
        kappa=1.0e-5,
        target_flat_range=(1.9440, 1.9465),
        target_normalized_intensity=0.2,
        default_iterations=100,
        b_initial=jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex64),
        delta_k2=jnp.array(50.0),
    )


# --- Helper Functions ---
def build_segment_domains(n_periods: int, period: float, kappa: float) -> tuple[jax.Array, jax.Array]:
    """Build domain widths and kappa values for a standard segment."""
    half_period = period / 2.0
    domain_widths = jnp.tile(jnp.array([half_period, half_period]), n_periods)
    kappa_vals = jnp.tile(jnp.array([kappa, -kappa]), n_periods)
    return domain_widths, kappa_vals


def build_phase_shift_domain(length: float, kappa: float) -> tuple[jax.Array, jax.Array]:
    """Build domain for a phase shift segment (fixed kappa = +1)."""
    return jnp.array([length]), jnp.array([kappa])


def build_paper_structure(cfg: SimConfig) -> tuple[jax.Array, jax.Array]:
    """Build the complete 3-segment aperiodic QPM grating structure."""
    n1 = 425  # Segment 1 periods
    n2 = 2245  # Segment 2 periods
    n3 = 425  # Segment 3 periods
    delta1 = 1.28  # Phase shift 1 length [µm]
    delta2 = 1.95  # Phase shift 2 length [µm]

    seg1_widths, seg1_kappa = build_segment_domains(n1, cfg.grating_period, cfg.kappa)
    ps1_widths, ps1_kappa = build_phase_shift_domain(delta1, cfg.kappa)
    seg2_widths, seg2_kappa = build_segment_domains(n2, cfg.grating_period, cfg.kappa)
    ps2_widths, ps2_kappa = build_phase_shift_domain(delta2, cfg.kappa)
    seg3_widths, seg3_kappa = build_segment_domains(n3, cfg.grating_period, cfg.kappa)

    domain_widths = jnp.concatenate([seg1_widths, ps1_widths, seg2_widths, ps2_widths, seg3_widths])
    kappa_vals = jnp.concatenate([seg1_kappa, ps1_kappa, seg2_kappa, ps2_kappa, seg3_kappa])
    return domain_widths, kappa_vals


def save_checkpoint(filename, params, opt_state):
    """Saves params and optimizer state to a pickle file."""
    data = {"params": params, "opt_state": opt_state}
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"State saved to '{filename}'")


def load_checkpoint(filename):
    """Loads params and optimizer state from a pickle file."""
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"State loaded from '{filename}'")
    return data["params"], data["opt_state"]


def make_optimization_step(optimizer, dk_targets, kappa_vals, delta_k2, b_initial, max_intensity_ref, target_norm_intensity, iterations):
    """Creates the optimization scan function using Amplitude-based MSE."""

    # Pre-calculate amplitude targets to avoid re-computing every step
    # We compare |A| vs |A|, rather than |A|^2 vs |A|^2
    max_amp_ref = jnp.sqrt(max_intensity_ref)
    target_amp_norm = jnp.sqrt(target_norm_intensity)

    # vmap definition: returns (Batch, 3)
    batch_sim = jax.vmap(cwes2.simulate_twm, in_axes=(None, None, None, 0, None, None))

    def loss_fn(params_widths):
        # Enforce absolute value on widths during simulation
        real_widths = jnp.abs(params_widths)

        # Get (Batch, 3) results
        b_batch = batch_sim(real_widths, kappa_vals, kappa_vals, dk_targets, delta_k2, b_initial)

        # Extract Component 1 (Signal/SHG) -> (Batch,)
        amps = b_batch[:, 1]

        # Normalize Amplitude
        norm_amp = jnp.abs(amps) / max_amp_ref

        # Calculate MSE on Amplitudes (Linear dependence creates better gradients)
        return jnp.mean((norm_amp - target_amp_norm) ** 2)

    @jax.jit
    @scan_tqdm(iterations, print_rate=10)
    def step_scan(state, i):
        params, opt_state = state
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params, value=loss_val, grad=grads, value_fn=loss_fn)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss_val

    return step_scan


def plot_results(
    dk_scan: jax.Array,
    ref_int: jax.Array,
    opt_int: jax.Array,
    loss_hist: jax.Array,
    target_range: tuple[float, float],
    target_intensity: float,
) -> None:
    """Generates and saves the optimization result plots."""

    # Spectrum Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dk_scan, ref_int, "--", color="gray", label="Uniform (Reference)")
    plt.plot(dk_scan, opt_int, "-", color="#2E86AB", label="Optimized")
    plt.axvspan(*target_range, color="g", alpha=0.1, label="Target Range")
    plt.hlines(target_intensity, *target_range, "r", "--", label="Target Level")
    plt.legend()
    plt.title("PPLN Optimization Result")
    plt.xlabel(r"$\Delta k$")
    plt.ylabel("Normalized Intensity")
    plt.savefig("ppln_optimized.png")
    plt.close()

    # Loss History Plot
    plt.figure(figsize=(10, 6))
    plt.plot(jnp.arange(len(loss_hist)), loss_hist, "-", color="#E63946")
    plt.yscale("log")
    plt.title("Optimization Loss History (Amplitude MSE)")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig("loss_history.png")
    plt.close()


def main():
    # --- Config ---
    cfg = get_default_config()

    # --- CLI Arguments ---
    parser = argparse.ArgumentParser(description="PPLN Optimization with Resume Capability")
    parser.add_argument("--resume", type=str, default=None, help="Path to .pkl file to resume from.")
    parser.add_argument("--save", type=str, default="checkpoint.pkl", help="Path to save the result .pkl file.")
    parser.add_argument("--iters", type=int, default=cfg.default_iterations, help="Number of iterations to run.")
    args = parser.parse_args()

    # --- Setup ---
    # key = jax.random.PRNGKey(42)  # Not using random noise for initial guess anymore

    # Build 3-segment initial structure
    print("Building 3-segment initial structure...")
    initial_widths, kappa_vals = build_paper_structure(cfg)

    # We also need a uniform structure for reference normalization
    total_length = jnp.sum(initial_widths)
    # Estimate equivalent periods for uniform grating
    num_uniform_periods = int(round(total_length / cfg.grating_period))

    half_p = cfg.grating_period / 2.0
    uniform_widths = jnp.tile(jnp.array([half_p, half_p]), num_uniform_periods)
    # Uniform kappa vals for reference (strictly periodic)
    uniform_kappas = jnp.tile(jnp.array([cfg.kappa, -cfg.kappa]), num_uniform_periods)

    dk_center = 2.0 * jnp.pi / cfg.grating_period

    # --- Normalization: Find Max from Uniform Distribution ---
    print("Calculating reference maximum intensity...")
    batch_sim = jax.vmap(cwes2.simulate_twm, in_axes=(None, None, None, 0, None, None))
    dk_ref_scan = jnp.linspace(dk_center * 0.99, dk_center * 1.01, 500)

    # Note: Use uniform_kappas for the reference simulation
    ref_b_batch_scan = batch_sim(uniform_widths, uniform_kappas, uniform_kappas, dk_ref_scan, cfg.delta_k2, cfg.b_initial)
    max_intensity = jnp.max(jnp.abs(ref_b_batch_scan[:, 1]) ** 2)
    print(f"Max Intensity (Reference): {max_intensity:.4e}")

    # --- Optimizer Init ---
    print(f"\nSetting up L-BFGS (Iterations: {args.iters})...")
    optimizer = optax.lbfgs(learning_rate=1.0)
    dk_targets = jnp.linspace(*cfg.target_flat_range, 50)

    # --- State Initialization (New or Resume) ---
    if args.resume and os.path.exists(args.resume):
        # Resume
        current_params, current_opt_state = load_checkpoint(args.resume)
    else:
        # New Start
        if args.resume:
            print(f"Warning: Checkpoint '{args.resume}' not found. Starting fresh.")

        # Use the 3-segment structure as initial parameters
        current_params = initial_widths
        current_opt_state = optimizer.init(current_params)
        print("Initialized parameters from 3-segment structure.")

    # --- Optimization Loop ---
    # Pass raw intensity values; helper function converts to amplitude
    step_fn = make_optimization_step(
        optimizer,
        dk_targets,
        kappa_vals,
        cfg.delta_k2,
        cfg.b_initial,
        max_intensity,
        cfg.target_normalized_intensity,
        args.iters,
    )

    print("Starting optimization scan...")
    (final_params, final_opt_state), loss_hist = jax.lax.scan(step_fn, (current_params, current_opt_state), jnp.arange(args.iters))

    # Ensure widths are positive for final output
    final_params_abs = jnp.abs(final_params)
    print(f"Final Loss: {loss_hist[-1]:.6e}")

    # --- Save Checkpoint ---
    save_checkpoint(args.save, final_params, final_opt_state)

    # --- Plotting ---
    print("Generating plots...")
    dk_scan = jnp.linspace(dk_center * 0.999, dk_center * 1.001, 1000)

    # Reference Spectrum
    ref_b_batch = batch_sim(uniform_widths, uniform_kappas, uniform_kappas, dk_scan, cfg.delta_k2, cfg.b_initial)
    ref_int = (jnp.abs(ref_b_batch[:, 1]) ** 2) / max_intensity

    # Optimized Spectrum
    opt_b_batch = batch_sim(final_params_abs, kappa_vals, kappa_vals, dk_scan, cfg.delta_k2, cfg.b_initial)
    opt_int = (jnp.abs(opt_b_batch[:, 1]) ** 2) / max_intensity

    plot_results(dk_scan, ref_int, opt_int, loss_hist, cfg.target_flat_range, cfg.target_normalized_intensity)
    print("Done.")


if __name__ == "__main__":
    main()
