import argparse
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax_tqdm import scan_tqdm  # Added jax_tqdm  # pyright: ignore[reportPrivateImportUsage]

from qpm import cwes2, mgoslt

jax.config.update("jax_enable_x64", val=True)


def get_static_constants(n_domains: int):
    """Pre-calculates physics constants and k-vectors."""
    wavelength = 1.064
    temp = 70.0

    dk1 = float(mgoslt.calc_twm_delta_k(wavelength, wavelength, temp))
    dk2 = float(mgoslt.calc_twm_delta_k(wavelength, wavelength / 2, temp))
    amp_fund = jnp.sqrt(10.0)
    b_init = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)

    k_val_shg = 1.5e-5 / (2 / np.pi)
    k_val_sfg = 1.5e-5 / (2 / np.pi) * 2

    s = jnp.tile(jnp.array([1.0, -1.0]), (n_domains // 2 + 1))[:n_domains]
    k_shg = s * k_val_shg
    k_sfg = s * k_val_sfg

    return dk1, dk2, b_init, k_shg, k_sfg


@partial(jax.jit, static_argnames=["iters", "bs", "min_width"])
def run_optimization(w_init, dk1, dk2, b_init, k_shg, k_sfg, iters, bs, min_width, penalty_w):
    """
    JIT-compiled optimization loop using Optax L-BFGS with TQDM progress bar.
    """
    solver = optax.lbfgs(learning_rate=1.0)
    opt_state = solver.init(w_init)

    def loss_fn(widths):
        w_real = jnp.abs(widths)
        # Physics Simulation
        b_final = cwes2.simulate_super_step(w_real, k_shg, k_sfg, dk1, dk2, b_init, bs)
        obj = -(jnp.abs(b_final[2]) ** 2)

        # Constraints
        if min_width is not None:
            violation = jax.nn.relu(min_width - w_real)
            obj += penalty_w * jnp.sum(violation**2)

        return obj

    # --- TQDM Decorator Applied Here ---
    @scan_tqdm(iters, print_rate=100)
    def step(carry, i):
        params, state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = solver.update(grads, state, params, value=loss, grad=grads, value_fn=loss_fn)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_state), loss

    # Scan over the iterations
    (final_w, _), loss_history = jax.lax.scan(step, (w_init, opt_state), jnp.arange(iters))

    return final_w, loss_history


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_pkl", nargs="?", default="best_polished.pkl")
    parser.add_argument("--iters", type=int, default=2000, help="Optimization steps")
    parser.add_argument("--min-width", type=float, default=None, help="Min domain width (microns)")
    parser.add_argument("--penalty", type=float, default=100.0, help="Constraint penalty weight")
    parser.add_argument("--output", type=str, default="best_optax_tqdm.pkl")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading {args.input_pkl}...")
    try:
        with open(args.input_pkl, "rb") as f:
            data = pickle.load(f)
            w_init = jnp.array(data["widths"])
            amp_seed = data["amp"]
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    n_domains = len(w_init)

    # 2. Setup
    dk1, dk2, b_init, k_shg, k_sfg = get_static_constants(n_domains)
    bs = next((b for b in range(min(300, n_domains), 19, -1) if n_domains % b == 0), 300)
    print(f"Config: {n_domains} domains | Block Size: {bs} | L-BFGS Iters: {args.iters}")

    # 3. Run Optimization
    print("Compiling and running optimizer...")
    t0 = time.time()

    final_w, history = run_optimization(w_init, dk1, dk2, b_init, k_shg, k_sfg, args.iters, bs, args.min_width, args.penalty)

    # Force sync
    final_w.block_until_ready()
    duration = time.time() - t0

    # 4. Results
    final_amp = float(jnp.abs(cwes2.simulate_super_step(jnp.abs(final_w), k_shg, k_sfg, dk1, dk2, b_init, bs)[2]))

    print(f"Completed in {duration:.2f}s")
    print(f"Amplitude: {amp_seed:.6f} -> {final_amp:.6f}")

    if final_amp > amp_seed:
        with open(args.output, "wb") as f:
            pickle.dump({"widths": np.array(jnp.abs(final_w)), "amp": final_amp}, f)
        print(f"Saved optimized result to {args.output}")

        plt.plot(history)
        plt.title("L-BFGS Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(args.output.replace(".pkl", "_loss.png"))
    else:
        print("No improvement observed.")


if __name__ == "__main__":
    main()
