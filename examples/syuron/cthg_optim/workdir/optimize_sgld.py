import argparse
import pickle
import time

import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from qpm import cwes2, mgoslt

# Constants
WAVELENGTH = 1.064
TEMPERATURE = 70.0


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


def compute_min_width_penalty(real_widths, min_width, penalty_scale=100.0, penalty_weight=10000.0):
    """Compute penalty for minimum width constraint violation."""
    if min_width is None:
        return 0.0
    violation = jax.nn.softplus(penalty_scale * (min_width - real_widths)) / penalty_scale
    return penalty_weight * jnp.sum(violation**2)


def make_logprob_fn(temperature_beta, min_width, block_size, k_val_shg, k_val_sfg, dk1, dk2, b_init):
    """Create log probability function for SGLD optimization."""

    def logprob(params_widths, batch=None):
        real_widths = jnp.abs(params_widths)
        n = real_widths.shape[0]
        k_shg, k_sfg = compute_signs_and_k_values(n, k_val_shg, k_val_sfg, sign_multiplier=1.0)

        b_final = cwes2.simulate_super_step(real_widths, k_shg, k_sfg, dk1, dk2, b_init, block_size)
        b3 = b_final[2]

        amp2 = jnp.abs(b3) ** 2
        penalty = compute_min_width_penalty(real_widths, min_width)

        return temperature_beta * amp2 - penalty

    return logprob


def evaluate_amplitude(widths, n_domains, k_val_shg, k_val_sfg, dk1, dk2, b_init, block_size):
    """Evaluate amplitude for given widths."""
    w_real = jnp.abs(widths)
    k_shg, k_sfg = compute_signs_and_k_values(n_domains, k_val_shg, k_val_sfg, sign_multiplier=1.0)
    b_final = cwes2.simulate_super_step(w_real, k_shg, k_sfg, dk1, dk2, b_init, block_size)
    return jnp.abs(b_final[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pkl", type=str, help="Path to the .pkl file containing initial widths")
    parser.add_argument("--steps", type=int, default=30000, help="Number of SGLD steps")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--temp", type=float, default=100.0, help="Temperature parameter")
    parser.add_argument(
        "--min-width",
        type=float,
        default=None,
        help="Minimum width constraint (optional, default: no constraint)",
    )
    args = parser.parse_args()

    # Physics setup
    dk1, dk2, k_val_shg, k_val_sfg, b_init = compute_physics_params()

    # Load initialization
    print(f"Loading seed from {args.input_pkl}...")
    with open(args.input_pkl, "rb") as f:
        data = pickle.load(f)
        w_init = data["widths"]
        print(f"Seed Amplitude: {data.get('amp', 'N/A')}")

    n = len(w_init)
    print(f"Initial domains: {n}")

    # Block size computation
    bs = compute_block_size(n)
    print(f"Block size: {bs}")

    # Minimum width constraint
    if args.min_width is not None:
        print(f"Minimum width constraint: {args.min_width}")
    else:
        print("No minimum width constraint")

    # Setup log probability function and SGLD algorithm
    logprob_fn = make_logprob_fn(args.temp, args.min_width, bs, k_val_shg, k_val_sfg, dk1, dk2, b_init)
    alg = blackjax.sgld(jax.grad(logprob_fn))

    initial_position = jnp.array(w_init)
    state = alg.init(initial_position)

    @jax.jit
    def one_step(state, key):
        state = alg.step(key, state, None, args.lr)
        return state, state

    # Run optimization
    print(f"Running SGLD for {args.steps} steps...")
    rng_key = jax.random.PRNGKey(int(time.time()))
    keys = jax.random.split(rng_key, args.steps)

    _, positions_history = jax.lax.scan(one_step, state, keys)

    # Evaluation
    positions_subs = positions_history[::100]
    print(f"Evaluating trace ({len(positions_subs)} samples)...")

    def get_amp(w):
        return evaluate_amplitude(w, n, k_val_shg, k_val_sfg, dk1, dk2, b_init, bs)

    amps = jax.lax.map(get_amp, positions_subs)

    best_idx = jnp.argmax(amps)
    best_amp = float(amps[best_idx])
    print(f"Best Amplitude: {best_amp:.6f}")

    # Save results
    out_name = args.input_pkl.replace(".pkl", "")
    filename = f"{out_name}_optimized.pkl"

    best_w = positions_subs[best_idx]
    with open(filename, "wb") as f:
        pickle.dump({"widths": np.array(best_w), "amp": best_amp}, f)
    print(f"Saved best result to {filename}")

    # Plot trace
    plt.figure()
    plt.plot(amps)
    plt.title(f"SGLD Trace ({out_name})")
    plt.savefig(f"{out_name}_trace.png")


if __name__ == "__main__":
    main()
