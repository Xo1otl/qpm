import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", val=True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax_tqdm import scan_tqdm  # pyright: ignore[reportPrivateImportUsage]

from qpm import cwes2, mgoslt


@dataclass
class SimulationConfig:
    shg_len: float = 18000
    sfg_len: float = 0
    kappa_shg: float = 1.5e-5 / (2 / jnp.pi)
    kappa_sfg: float = 1.5e-5 / (2 / jnp.pi) * 2
    temperature: float = 70.0
    wavelength: float = 1.064
    input_power: float = 10.0  # NOTE: initial amplitude is sqrt of this power
    block_size: int = 300
    iterations: int = 500


def get_initial_structure(cfg: SimulationConfig) -> tuple[jax.Array, jax.Array, jax.Array, tuple[float, float, float, float]]:
    """
    Constructs the initial tandem structure (SHG + SFG).
    Returns:
        widths, kappa_shg_vals, kappa_sfg_vals, (dk1, dk2, lc1, lc2)
    """
    # Calculate wave vectors and mismatch
    dk_shg = mgoslt.calc_twm_delta_k(cfg.wavelength, cfg.wavelength, cfg.temperature)
    # SFG: w + 2w -> 3w. Inputs are wl, wl/2.
    dk_sfg = mgoslt.calc_twm_delta_k(cfg.wavelength, cfg.wavelength / 2, cfg.temperature)

    # Coherence lengths
    lc_shg = jnp.pi / dk_shg
    lc_sfg = jnp.pi / dk_sfg

    # --- Build SHG Section ---
    # Period = 2 * Lc
    period_shg = 2 * lc_shg
    n_periods_shg = round(cfg.shg_len / period_shg)

    # 2 domains per period
    widths_shg = jnp.tile(jnp.array([lc_shg, lc_shg]), n_periods_shg)
    # Signs flip: +1, -1, +1, -1...
    signs_shg = jnp.tile(jnp.array([1.0, -1.0]), n_periods_shg)

    # --- Build SFG Section ---
    period_sfg = 2 * lc_sfg
    n_periods_sfg = round(cfg.sfg_len / period_sfg)

    widths_sfg = jnp.tile(jnp.array([lc_sfg, lc_sfg]), n_periods_sfg)
    signs_sfg = jnp.tile(jnp.array([1.0, -1.0]), n_periods_sfg)

    # --- Combine ---
    widths = jnp.concatenate([widths_shg, widths_sfg])
    signs = jnp.concatenate([signs_shg, signs_sfg])

    # Kappa arrays
    # Note: The sign of the domain affects BOTH processes in the same way (inverted domain).
    k_shg_vals = signs * cfg.kappa_shg
    k_sfg_vals = signs * cfg.kappa_sfg

    return widths, k_shg_vals, k_sfg_vals, (dk_shg, dk_sfg, lc_shg, lc_sfg)


def make_optimization_step(
    optimizer: optax.GradientTransformation,
    kappa_shg_vals: jax.Array,
    kappa_sfg_vals: jax.Array,
    delta_k1: jax.Array,
    delta_k2: jax.Array,
    b_initial: jax.Array,
    iterations: int,
    block_size: int,
):
    """Creates the JIT-compiled optimization scan loop."""

    # Loss function: Maximize |A3|.
    # We define loss = -|A3|^2 / P_in (or just -|A3|).
    # Let's use negative squared absolute value (Intensity) as it's smooth.

    def loss_fn(params_widths):
        # Enforce physical constraints (widths > 0)
        real_widths = jnp.abs(params_widths)

        # Simulate
        # Note: simulate_super_step returns the final B vector: [B1, B2, B3]
        b_final = cwes2.simulate_super_step(real_widths, kappa_shg_vals, kappa_sfg_vals, delta_k1, delta_k2, b_initial, block_size)

        # B3 is the THG wave
        b3 = b_final[2]

        # We want to maximize intensity |B3|^2 -> minimize -|B3|^2
        return -(jnp.abs(b3) ** 2)

    @jax.jit
    @scan_tqdm(iterations, print_rate=10)
    def step_scan(state, i):
        params, opt_state = state
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params, value=loss_val, grad=grads, value_fn=loss_fn)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss_val

    return step_scan, loss_fn


def run_optimization(
    cfg: SimulationConfig,
    initial_widths: jax.Array,
    kappa_shg: jax.Array,
    kappa_sfg: jax.Array,
    dk1: jax.Array,
    dk2: jax.Array,
    b_init: jax.Array,
    iterations: int,
    initial_opt_state: optax.OptState | None = None,
):
    print(f"\nSetting up L-BFGS (Iterations: {iterations})...")
    optimizer = optax.lbfgs(learning_rate=1.0)  # L-BFGS usually needs lr=1.0

    current_params = initial_widths

    current_opt_state = optimizer.init(current_params) if initial_opt_state is None else initial_opt_state

    step_fn, internal_loss_fn = make_optimization_step(optimizer, kappa_shg, kappa_sfg, dk1, dk2, b_init, iterations, cfg.block_size)

    print("Starting optimization scan...")
    (final_params, final_opt_state), loss_hist = jax.lax.scan(step_fn, (current_params, current_opt_state), jnp.arange(iterations))

    return final_params, final_opt_state, loss_hist, internal_loss_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=500, help="Number of iterations")
    parser.add_argument("--save", type=str, default="amp_opt_result.pkl", help="Save path")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint pkl to resume from")
    args = parser.parse_args()

    cfg = SimulationConfig(iterations=args.iters)

    print("--- Configuration ---")
    print(cfg)

    # 1. Setup Structure
    # If resuming, we load the structure from the file, but we usually recompute the environment (dk, kappa magnitude)
    # to be safe, or we could load config. Let's assume the user runs with same config or we re-generate 'widths' logic just to get kappas right.

    # We always need the kappa/dk context.
    dummy_widths, k_shg, k_sfg, (dk1, dk2, lc1, lc2) = get_initial_structure(cfg)

    print(f"Delta k1 (SHG): {dk1:.4f} (Lc = {lc1:.2f} um)")
    print(f"Delta k2 (SFG): {dk2:.4f} (Lc = {lc2:.2f} um)")

    # 2. Initial State
    amp_fund = jnp.sqrt(cfg.input_power)
    b_initial = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)

    initial_opt_state = None

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        with Path(args.resume).open("rb") as f:
            checkpoint = pickle.load(f)

        # Load params and state
        widths = checkpoint["params"]
        initial_opt_state = checkpoint.get("opt_state")

        # Load config if available to ensure physics match
        if "config" in checkpoint:
            saved_cfg = checkpoint["config"]
            # basic update of matching fields
            if isinstance(saved_cfg, dict):
                # Update cfg with saved values
                for k, v in saved_cfg.items():
                    if hasattr(cfg, k) and k != "iterations":
                        setattr(cfg, k, v)
                print("Loaded configuration from checkpoint.")
            elif hasattr(saved_cfg, "__dict__"):
                for k, v in saved_cfg.__dict__.items():
                    if hasattr(cfg, k) and k != "iterations":
                        setattr(cfg, k, v)
                print("Loaded configuration from checkpoint object.")

            # Re-fetch structure constants with potentially updated config
            _, _, _, (dk1, dk2, lc1, lc2) = get_initial_structure(cfg)
            print(f"Updated Delta k1: {dk1:.4f}, Delta k2: {dk2:.4f}")

        if initial_opt_state is None:
            print("Checkpoint contains no optimizer state. Starting fresh optimizer.")

        print(f"Loaded {len(widths)} domains from checkpoint.")

        # Regenerate kappa arrays for the resizes structure
        n_domains = len(widths)
        signs = jnp.ones(n_domains)
        signs = signs.at[1::2].set(-1.0)
        k_shg = signs * cfg.kappa_shg
        k_sfg = signs * cfg.kappa_sfg
    else:
        widths = dummy_widths
        print(f"\nInitial Structure: {len(widths)} domains")

    # Use physical widths (absolute value) for reporting and initial simulation
    physical_widths = jnp.abs(widths)
    print(f"Total Length: {jnp.sum(physical_widths):.2f} um")

    # Dynamic Block Size Calculation
    n_domains = len(widths)
    # Default from config or args?
    # Let's try to fine a divisor >= 20
    dynamic_bs = cfg.block_size

    # Try to find a good divisor
    for bs in range(200, 19, -1):
        if n_domains % bs == 0:
            dynamic_bs = bs
            break

    print(f"Using Dynamic Block Size: {dynamic_bs}")

    # Note: make_optimization_step needs to be called with this new block size
    # But wait, run_optimization calls make_optimization_step.
    # We need to update cfg.block_size or pass it.
    cfg.block_size = dynamic_bs

    # 3. Baseline Check
    b_final_init = cwes2.simulate_super_step(physical_widths, k_shg, k_sfg, dk1, dk2, b_initial, block_size=dynamic_bs)
    amp_thg_start = jnp.abs(b_final_init[2])
    print(f"\nStart THG Amplitude: {amp_thg_start:.6f}")
    print(f"Start THG Intensity: {amp_thg_start**2:.6f}")

    # 4. Optimization

    final_params, final_opt_state, loss_hist, _ = run_optimization(
        cfg,
        widths,
        k_shg,
        k_sfg,
        dk1,
        dk2,
        b_initial,
        cfg.iterations,
        initial_opt_state=initial_opt_state,
    )

    # 5. Results
    final_widths = jnp.abs(final_params)
    b_final_opt = cwes2.simulate_super_step(final_widths, k_shg, k_sfg, dk1, dk2, b_initial)
    thg_amp_opt = jnp.abs(b_final_opt[2])

    print(f"\nOptimized THG Amplitude: {thg_amp_opt:.6f}")
    print(f"Optimized THG Intensity: {thg_amp_opt**2:.6f}")
    print(f"Total Length (Optimized): {jnp.sum(final_widths):.2f} um")
    print(f"Improvement Factor: {thg_amp_opt**2 / (amp_thg_start**2):.4f}x")

    # Save results as pickle
    checkpoint_data = {
        "params": final_params,
        "opt_state": final_opt_state,
        "config": cfg,
        "loss_hist": loss_hist,
        "thg_amp": thg_amp_opt,  # Trailing comma
    }

    with Path(args.save).open("wb") as f:
        pickle.dump(checkpoint_data, f)
    print(f"\nSaved results to {args.save}")

    # Save plot
    plt.plot(loss_hist)
    plt.savefig("loss_hist.png")


if __name__ == "__main__":
    main()
