"""Sep-CMA-ES optimization (Simplified) - No Repair, Pure Amplitude Maximization.

Usage:
    python optimize_cmaes_simple.py best_optimized_lbfgs.pkl --gens 200 --popsize 2048 --std-init 0.0001
"""

from __future__ import annotations

import argparse
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms.distribution_based.sep_cma_es import Sep_CMA_ES

from qpm import cwes2, mgoslt

jax.config.update("jax_enable_x64", val=True)

# Constants
WAVELENGTH = 1.064
TEMPERATURE = 70.0


def compute_physics_params():
    dk1 = float(mgoslt.calc_twm_delta_k(WAVELENGTH, WAVELENGTH, TEMPERATURE))
    dk2 = float(mgoslt.calc_twm_delta_k(WAVELENGTH, WAVELENGTH / 2, TEMPERATURE))
    k_val_shg = 1.5e-5 / (2 / np.pi)
    k_val_sfg = 1.5e-5 / (2 / np.pi) * 2
    amp_fund = jnp.sqrt(10.0)
    b_init = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)
    return dk1, dk2, k_val_shg, k_val_sfg, b_init


def compute_block_size(n_domains: int) -> int:
    bs = 300
    for b in range(min(500, n_domains), 19, -1):
        if n_domains % b == 0:
            bs = b
            break
    return bs


def compute_signs_and_k_values(n_domains: int, k_val_shg: float, k_val_sfg: float):
    s = jnp.tile(jnp.array([1.0, -1.0]), (n_domains // 2 + 1))[:n_domains]
    k_shg = s * k_val_shg
    k_sfg = s * k_val_sfg
    return k_shg, k_sfg


def evaluate_amplitude(
    widths: jax.Array,
    n_domains: int,
    k_val_shg: float,
    k_val_sfg: float,
    dk1: float,
    dk2: float,
    b_init: jax.Array,
    block_size: int,
) -> float:
    # Minimal safety: widths must be positive lengths
    w = jnp.abs(widths)
    k_shg, k_sfg = compute_signs_and_k_values(n_domains, k_val_shg, k_val_sfg)
    b_final = cwes2.simulate_super_step(w, k_shg, k_sfg, dk1, dk2, b_init, block_size)
    return float(jnp.abs(b_final[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplified Sep-CMA-ES (No Repair)")
    parser.add_argument("input_pkl", type=str, help="Path to seed .pkl")
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--popsize", type=int, default=1024)
    parser.add_argument("--std-init", type=float, default=0.0001, help="Small value for fine-tuning")
    parser.add_argument("--output", type=str, default="best_optimized_cmaes.pkl")
    args = parser.parse_args()

    # Load seed
    print(f"Loading seed from {args.input_pkl}...")
    with open(args.input_pkl, "rb") as f:
        data = pickle.load(f)
    w_init = np.array(data["widths"], dtype=np.float64)
    n = len(w_init)

    dk1, dk2, k_val_shg, k_val_sfg, b_init = compute_physics_params()
    block_size = compute_block_size(n)

    # Check initial amplitude
    amp_seed = evaluate_amplitude(jnp.array(w_init), n, k_val_shg, k_val_sfg, dk1, dk2, b_init, block_size)
    print(f"Seed amplitude: {amp_seed:.6f}, domains: {n}")

    # Setup Strategy
    # Using small std_init relative to typical width (approx 3.0um)
    # std=0.0001 means ~0.3nm perturbation, suitable for fine-tuning.
    w_scale = float(np.mean(np.abs(w_init)))
    std_init = args.std_init * w_scale

    solution = jnp.zeros(n)
    strategy = Sep_CMA_ES(population_size=args.popsize, solution=solution)
    params = strategy.default_params.replace(std_init=std_init)

    key = jax.random.PRNGKey(int(time.time()))
    mean_init = jnp.array(w_init)

    # Force Mean Initialization explicitly
    state = strategy.init(key, mean_init, params)
    state = state.replace(mean=mean_init)

    # Pre-compute constants for JIT
    k_shg, k_sfg = compute_signs_and_k_values(n, k_val_shg, k_val_sfg)

    @jax.jit
    def fit_fn(population):
        def sim_one(w):
            # Pure optimization: just absolute value to prevent math errors
            w_abs = jnp.abs(w)
            b_final = cwes2.simulate_super_step(w_abs, k_shg, k_sfg, dk1, dk2, b_init, block_size)
            # Minimize negative squared amplitude
            return -(jnp.abs(b_final[2]) ** 2)

        return jax.vmap(sim_one)(population)

    best_amp = amp_seed
    best_w = mean_init
    loss_hist = []

    print(f"Starting optimization: Gens={args.gens}, Pop={args.popsize}, Std={args.std_init}")

    for g in range(args.gens):
        key, k1, k2 = jax.random.split(key, 3)
        population, state = strategy.ask(k1, state, params)

        fitness = fit_fn(population)

        state, metrics = strategy.tell(k2, population, fitness, state, params)

        # Track best
        current_gen_best_idx = jnp.argmin(fitness)
        current_gen_best_fit = fitness[current_gen_best_idx]
        current_amp = jnp.sqrt(-current_gen_best_fit)

        loss_hist.append(float(current_gen_best_fit))

        if current_amp > best_amp:
            best_amp = float(current_amp)
            best_w = jnp.abs(population[current_gen_best_idx])  # Store physically valid widths

        if (g + 1) % 10 == 0 or g == 0:
            print(f" gen {g + 1}: best_amp={best_amp:.6f}  gen_best={current_amp:.6f}")

    print(f"Final amp: {best_amp:.6f} (Improved: {best_amp > amp_seed})")

    if best_amp > amp_seed:
        with open(args.output, "wb") as f:
            pickle.dump({"widths": np.array(best_w), "amp": best_amp}, f)
        print(f"Saved to {args.output}")

        # Quick validation of length drift
        total_len = np.sum(np.abs(best_w))
        print(f"Final Total Length: {total_len:.2f} um")
    else:
        print("No improvement found.")


if __name__ == "__main__":
    main()
