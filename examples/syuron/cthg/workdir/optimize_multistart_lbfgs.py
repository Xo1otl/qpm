import argparse
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from jax_tqdm import scan_tqdm  # pip install jax-tqdm

from qpm import cwes2, mgoslt

jax.config.update("jax_enable_x64", val=True)

# --- Constants ---
WAVELENGTH = 1.064
TEMPERATURE = 70.0
LR = 1.0


def get_physics_setup(n):
    """Returns physics constants and pre-computed K-vectors."""
    dk1 = float(mgoslt.calc_twm_delta_k(WAVELENGTH, WAVELENGTH, TEMPERATURE))
    dk2 = float(mgoslt.calc_twm_delta_k(WAVELENGTH, WAVELENGTH / 2, TEMPERATURE))
    k_base = 1.5e-5 / (2 / jnp.pi)

    # Pre-compute alternating K-vectors [1, -1, 1...]
    s = jnp.tile(jnp.array([1.0, -1.0]), (n // 2 + 1))[:n]
    return dk1, dk2, s * k_base, s * k_base * 2, jnp.array([jnp.sqrt(10.0), 0j, 0j])


def compute_block_size(n):
    c = jnp.arange(min(500, n), 19, -1)
    return int(jnp.max(jnp.where(n % c == 0, c, 0))) or 300


# --- Core Logic ---
def loss_fn(widths, k_shg, k_sfg, dk1, dk2, b_init, penalty, min_w, bs):
    w_real = jnp.abs(widths)
    b_final = cwes2.simulate_super_step(w_real, k_shg, k_sfg, dk1, dk2, b_init, bs)
    # return -(jnp.abs(b_final[2]) ** 2) + penalty * jnp.sum(jax.nn.relu(min_w - w_real) ** 2)
    return -(jnp.abs(b_final[2]) ** 2)


@partial(jax.jit, static_argnames=["iters", "bs"])
def optimize_all(init_widths, k_shg, k_sfg, dk1, dk2, b_init, penalty, min_w, iters, bs):
    opt = optax.lbfgs(learning_rate=LR)
    opt_states = jax.vmap(opt.init)(init_widths)

    @scan_tqdm(iters)
    def step(carry, _):
        params, states = carry

        # 1. Define single-instance loss for this step
        calc_loss = lambda p: loss_fn(p, k_shg, k_sfg, dk1, dk2, b_init, penalty, min_w, bs)

        # 2. Compute gradients for the batch
        vals, grads = jax.vmap(jax.value_and_grad(calc_loss))(params)

        # 3. Update Wrapper
        # We wrap opt.update so 'calc_loss' is captured by closure,
        # preventing vmap from seeing it as an argument it needs to batch.
        def single_update(g, s, p, v):
            return opt.update(g, s, p, value=v, grad=g, value_fn=calc_loss)

        # 4. Map the update
        updates, new_states = jax.vmap(single_update)(grads, states, params, vals)

        return (optax.apply_updates(params, updates), new_states), None

    # Pass jnp.arange(iters) so jax_tqdm tracks the loop count correctly
    (final_params, _), _ = jax.lax.scan(step, (init_widths, opt_states), jnp.arange(iters))

    # Final eval
    final_real = jnp.abs(final_params)
    final_amps = jax.vmap(lambda w: jnp.abs(cwes2.simulate_super_step(w, k_shg, k_sfg, dk1, dk2, b_init, bs)[2]))(final_real)

    return final_real, final_amps


def generate_starts(base, n_starts, min_w=1.5, seed=42):
    key = jr.PRNGKey(seed)
    n = base.shape[0]
    num_noise = n_starts - 1
    noise = jr.uniform(key, (num_noise, n), minval=-0.3, maxval=0.5)
    perturbed = jnp.maximum(base * (1 + noise), min_w)
    return jnp.vstack([base[None, :], perturbed])


# --- Main ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", default="best_ratio.pkl")
    p.add_argument("--n-starts", type=int, default=1000)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--penalty", type=float, default=0.0)
    p.add_argument("--output", default="best_multistart_2nd.pkl")
    args = p.parse_args()

    try:
        with open(args.input, "rb") as f:
            w_base = jnp.array(pickle.load(f)["widths"])
    except FileNotFoundError:
        return print("Input file not found.")

    dk1, dk2, k_shg, k_sfg, b_init = get_physics_setup(w_base.shape[0])
    bs = compute_block_size(w_base.shape[0])
    starts = generate_starts(w_base, args.n_starts)

    print(f"Optimizing {args.n_starts} parallel starts for {args.iters} iters...")
    t0 = time.time()

    final_w, final_a = optimize_all(starts, k_shg, k_sfg, dk1, dk2, b_init, args.penalty, 1.5, args.iters, bs)

    final_a.block_until_ready()
    best_idx = jnp.argmax(final_a)
    print(f"Done in {time.time() - t0:.2f}s. Best Amp: {final_a[best_idx]:.6f}")

    with open(args.output, "wb") as f:
        pickle.dump({"widths": final_w[best_idx], "amp": final_a[best_idx]}, f)

    plt.hist(final_a, bins=50)
    plt.title(f"Outcomes (Best: {final_a[best_idx]:.6f})")
    plt.savefig(args.output.replace(".pkl", ".png"))
    return None


if __name__ == "__main__":
    main()
