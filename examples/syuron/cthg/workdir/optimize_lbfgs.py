import argparse
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax_tqdm import scan_tqdm  # pyright: ignore[reportPrivateImportUsage]

from qpm import cwes2, mgoslt

jax.config.update("jax_enable_x64", val=True)

# Constants
WAVELENGTH = 1.064
TEMPERATURE = 70.0

# Global optimizer instance
optimizer = optax.lbfgs(learning_rate=1.0)


def compute_physics_params():
    """Compute physics parameters for the simulation."""
    dk1 = float(mgoslt.calc_twm_delta_k(WAVELENGTH, WAVELENGTH, TEMPERATURE))
    dk2 = float(mgoslt.calc_twm_delta_k(WAVELENGTH, WAVELENGTH / 2, TEMPERATURE))
    k_val_shg = 1.5e-5 / (2 / np.pi)
    k_val_sfg = 1.5e-5 / (2 / np.pi) * 2
    amp_fund = jnp.sqrt(10.0)
    b_init = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)
    return dk1, dk2, k_val_shg, k_val_sfg, b_init


def compute_block_size(n_domains):
    """Compute optimal block size for the given number of domains."""
    bs = 300
    for b in range(min(500, n_domains), 19, -1):
        if n_domains % b == 0:
            bs = b
            break
    return bs


def compute_signs_and_k_values(n_domains, k_val_shg, k_val_sfg, sign_multiplier=1.0):
    """Compute signs and k values for SHG and SFG."""
    s = jnp.tile(jnp.array([1.0, -1.0]), (n_domains // 2 + 1))[:n_domains]
    signs = s * sign_multiplier
    k_shg = signs * k_val_shg
    k_sfg = signs * k_val_sfg
    return k_shg, k_sfg


def compute_min_width_penalty(real_widths, min_width, penalty_weight=100.0):
    """Compute penalty for minimum width constraint violation."""
    if min_width is None:
        return 0.0
    violation = jax.nn.relu(min_width - real_widths)
    return penalty_weight * jnp.sum(violation**2)


def make_optimization_step(penalty_weight, min_width, block_size, k_val_shg, k_val_sfg, dk1, dk2, b_init, iterations):
    """Create optimization step function for L-BFGS."""

    def loss_fn(params_widths, initial_signs_ref):
        real_widths = jnp.abs(params_widths)
        n = real_widths.shape[0]
        k_shg, k_sfg = compute_signs_and_k_values(n, k_val_shg, k_val_sfg, sign_multiplier=initial_signs_ref)

        b_final = cwes2.simulate_super_step(real_widths, k_shg, k_sfg, dk1, dk2, b_init, block_size)
        b3 = b_final[2]

        obj = -(jnp.abs(b3) ** 2)
        penalty = compute_min_width_penalty(real_widths, min_width, penalty_weight)

        return obj + penalty

    @jax.jit
    @scan_tqdm(iterations, print_rate=1000)
    def step_scan(state, i):
        params, opt_state, sign_ref = state
        loss_val, grads = jax.value_and_grad(loss_fn)(params, sign_ref)
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params,
            value=loss_val,
            grad=grads,
            value_fn=lambda p: loss_fn(p, sign_ref),
        )
        params = optax.apply_updates(params, updates)
        return (params, opt_state, sign_ref), loss_val

    return step_scan


def evaluate_amplitude(widths, n_domains, k_val_shg, k_val_sfg, dk1, dk2, b_init, block_size):
    """Evaluate amplitude for given widths."""
    w_real = jnp.abs(widths)
    k_shg, k_sfg = compute_signs_and_k_values(n_domains, k_val_shg, k_val_sfg, sign_multiplier=1.0)
    b_final = cwes2.simulate_super_step(w_real, k_shg, k_sfg, dk1, dk2, b_init, block_size)
    return jnp.abs(b_final[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_pkl",
        type=str,
        nargs="?",
        default="best_polished.pkl",
        help="Path to the .pkl file containing initial widths (default: best_polished.pkl)",
    )
    parser.add_argument("--iters", type=int, default=10000, help="Number of L-BFGS iterations")
    parser.add_argument(
        "--penalty-weight",
        type=float,
        default=100.0,
        help="Weight for minimum width penalty (default: 100.0)",
    )
    parser.add_argument(
        "--min-width",
        type=float,
        default=None,
        help="Minimum width constraint (optional, default: no constraint)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="best_optimized_lbfgs.pkl",
        help="Output filename for improved results (default: best_polished_long.pkl)",
    )
    args = parser.parse_args()

    # Load initialization
    print(f"Loading result from {args.input_pkl}...")
    try:
        with open(args.input_pkl, "rb") as f:
            data = pickle.load(f)
            w_init = data["widths"]
            amp_seed = data["amp"]
            print(f"Seed Amplitude: {amp_seed}")
    except FileNotFoundError:
        print(f"Error: {args.input_pkl} not found.")
        return

    # Physics setup
    dk1, dk2, k_val_shg, k_val_sfg, b_init = compute_physics_params()

    w_jax = jnp.array(w_init)
    n = len(w_jax)

    # Block size computation
    bs = compute_block_size(n)
    print(f"Block size: {bs}")

    # Minimum width constraint
    if args.min_width is not None:
        print(f"Minimum width constraint: {args.min_width}")
    else:
        print("No minimum width constraint")

    # Run optimization
    print(f"Starting L-BFGS Polishing ({args.iters} steps)...")
    step_fn = make_optimization_step(
        args.penalty_weight,
        args.min_width,
        bs,
        k_val_shg,
        k_val_sfg,
        dk1,
        dk2,
        b_init,
        args.iters,
    )

    opt_state = optimizer.init(w_jax)
    start_sign = 1.0

    (final_widths, _, _), loss_hist = jax.lax.scan(step_fn, (w_jax, opt_state, start_sign), jnp.arange(args.iters))

    final_widths_np = np.array(jnp.abs(final_widths))

    # Evaluate final amplitude
    amp_opt = float(evaluate_amplitude(final_widths, n, k_val_shg, k_val_sfg, dk1, dk2, b_init, bs))

    print(f" Initial Amp: {amp_seed}")
    print(f" Polished Amp: {amp_opt:.6f}")

    if amp_opt > amp_seed:
        print(" Improved!")
        with open(args.output, "wb") as f:
            pickle.dump({"widths": final_widths_np, "amp": amp_opt}, f)
        print(f"Saved improved result to {args.output}")

        plt.figure()
        plt.plot(loss_hist)
        plt.title("L-BFGS Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        output_plot = args.output.replace(".pkl", "_loss.png")
        plt.savefig(output_plot)
        print(f"Saved loss plot to {output_plot}")
    else:
        print(" No improvement.")


if __name__ == "__main__":
    main()
