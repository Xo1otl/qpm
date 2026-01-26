import argparse
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

from qpm import cwes2, mgoslt

jax.config.update("jax_enable_x64", val=True)

# --- Constants ---
WAVELENGTH = 1.064
TEMPERATURE = 70.0
LEARNING_RATE = 1.0


def get_physics_constants():
    dk1 = float(mgoslt.calc_twm_delta_k(WAVELENGTH, WAVELENGTH, TEMPERATURE))
    dk2 = float(mgoslt.calc_twm_delta_k(WAVELENGTH, WAVELENGTH / 2, TEMPERATURE))
    k_shg = 1.5e-5 / (2 / jnp.pi)
    k_sfg = k_shg * 2
    b_init = jnp.array([jnp.sqrt(10.0), 0.0, 0.0], dtype=jnp.complex128)
    return dk1, dk2, k_shg, k_sfg, b_init


def compute_block_size(n_domains):
    max_b = min(500, n_domains)
    candidates = jnp.arange(max_b, 19, -1)
    bs = jnp.max(jnp.where(n_domains % candidates == 0, candidates, 0))
    return int(bs) if bs > 0 else 300


# --- Core Logic ---


def loss_fn(widths, sign_ref, constants, penalty_weight, min_width, block_size):
    dk1, dk2, k_val_shg, k_val_sfg, b_init = constants
    real_widths = jnp.abs(widths)
    n = real_widths.shape[0]

    # K vectors
    s = jnp.tile(jnp.array([1.0, -1.0]), (n // 2 + 1))[:n]
    signs = s * sign_ref
    k_shg = signs * k_val_shg
    k_sfg = signs * k_val_sfg

    # Simulation
    b_final = cwes2.simulate_super_step(real_widths, k_shg, k_sfg, dk1, dk2, b_init, block_size)

    # Objective: Maximize Output Intensity (Minimize negative)
    obj = -(jnp.abs(b_final[2]) ** 2)

    # Penalty: Minimum Width
    violation = jax.nn.relu(min_width - real_widths)
    penalty = penalty_weight * jnp.sum(violation**2)

    return obj + penalty


@partial(jax.jit, static_argnames=["iters", "block_size"])
def optimize_batch(initial_widths_batch, constants, penalty_weight, min_width, iters, block_size):
    optimizer = optax.lbfgs(learning_rate=LEARNING_RATE)

    def single_opt_run(init_w):
        opt_state = optimizer.init(init_w)
        sign_ref = 1.0

        def step(state, _):
            params, opt_st = state
            value, grads = jax.value_and_grad(loss_fn)(params, sign_ref, constants, penalty_weight, min_width, block_size)
            updates, opt_st = optimizer.update(
                grads,
                opt_st,
                params,
                value=value,
                grad=grads,
                value_fn=lambda p: loss_fn(p, sign_ref, constants, penalty_weight, min_width, block_size),
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_st), value

        (final_params, _), _ = jax.lax.scan(step, (init_w, opt_state), None, length=iters)

        # Final Evaluation (Pure physics, no penalty)
        final_real = jnp.abs(final_params)
        dk1, dk2, k_shg, k_sfg, b_init = constants
        s = jnp.tile(jnp.array([1.0, -1.0]), (final_real.shape[0] // 2 + 1))[: final_real.shape[0]]
        b_final = cwes2.simulate_super_step(final_real, s * k_shg, s * k_sfg, dk1, dk2, b_init, block_size)
        return final_real, jnp.abs(b_final[2])

    return jax.vmap(single_opt_run)(initial_widths_batch)


def generate_initial_values(base_widths, n_starts, min_width=1.5, seed=42):
    """
    Generates batch where Index 0 is the exact input (Polisher)
    and Indices 1+ are perturbed variations (Explorer).
    """
    key = jr.PRNGKey(seed)
    n = base_widths.shape[0]

    # Create noise for (n_starts - 1) items
    # We use a slightly smaller batch for noise generation, then append base
    k1, k2, k3 = jr.split(key, 3)
    num_noisy = n_starts - 1

    # Noise Strategies
    noise_g = jr.normal(k1, (num_noisy, n)) * 0.1
    noise_u = jr.uniform(k2, (num_noisy, n), minval=-0.2, maxval=0.2)
    scales = jr.uniform(k3, (num_noisy, 1), minval=0.95, maxval=1.05)

    # Mix strategies (60% Gaussian, 20% Uniform, 20% Scale)
    selector = jr.choice(key, jnp.arange(3), shape=(num_noisy, 1), p=jnp.array([0.6, 0.2, 0.2]))
    combined_noise = jnp.where(selector == 0, noise_g, jnp.where(selector == 1, noise_u, 0.0))

    # Apply noise
    base_tiled = jnp.tile(base_widths, (num_noisy, 1))
    perturbed = base_tiled * (1 + combined_noise)
    perturbed = jnp.where(selector == 2, perturbed * scales, perturbed)

    # Clamp perturbed values to strict feasibility
    perturbed = jnp.maximum(perturbed, min_width)

    # Combine: [Base (Raw)] + [Perturbed (Clamped)]
    # We leave Base raw to match Script A's behavior exactly (allowing penalty to guide it)
    final_batch = jnp.vstack([base_widths[None, :], perturbed])

    return final_batch


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pkl", nargs="?", default="best_ratio.pkl")
    parser.add_argument("--n-starts", type=int, default=100)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--penalty-weight", type=float, default=100.0)
    parser.add_argument("--min-width", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--output", default="best_multistart_lbfgs.pkl")
    args = parser.parse_args()

    try:
        with open(args.input_pkl, "rb") as f:
            data = pickle.load(f)
        w_base = jnp.array(data["widths"])
        print(f"Loaded {len(w_base)} domains. Base Amp: {data.get('amp', 0.0):.6f}")
    except FileNotFoundError:
        print("Input file not found.")
        return

    constants = get_physics_constants()
    bs = compute_block_size(w_base.shape[0])

    print("Generating initial values...")
    init_batch = generate_initial_values(w_base, args.n_starts, args.min_width)

    best_amp = -1.0
    best_widths = None
    all_results = []

    print(f"Starting optimization: {args.n_starts} starts, {args.iters} iters")
    start_time = time.time()

    num_batches = int(jnp.ceil(args.n_starts / args.batch_size))

    for i in tqdm(range(num_batches)):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, args.n_starts)
        current_batch = init_batch[start_idx:end_idx]

        batch_widths, batch_amps = optimize_batch(current_batch, constants, args.penalty_weight, args.min_width, args.iters, bs)

        # Check for new best
        batch_max = float(jnp.max(batch_amps))
        if batch_max > best_amp:
            idx = jnp.argmax(batch_amps)
            best_amp = batch_max
            best_widths = batch_widths[idx]
            print(f" New Best: {best_amp:.6f}")

        all_results.extend([float(a) for a in batch_amps])

    print(f"Total time: {time.time() - start_time:.2f}s")

    with open(args.output, "wb") as f:
        pickle.dump({"widths": best_widths, "amp": best_amp}, f)

    plt.figure(figsize=(10, 6))
    plt.hist(all_results, bins=50)
    plt.title(f"Distribution of Outcomes (Best: {best_amp:.6f})")
    plt.savefig(args.output.replace(".pkl", ".png"))
    print("Done.")


if __name__ == "__main__":
    main()
