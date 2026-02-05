import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax
import numpy as np
from qpm import cwes2, mgoslt
import time
import pickle
import argparse
from jax_tqdm import scan_tqdm  # pyright: ignore[reportPrivateImportUsage]

def main():
    parser = argparse.ArgumentParser(description="Continue optimization from a saved pickle file.")
    parser.add_argument("--file", type=str, default="best_solution_grad.pkl", help="Path to the pickle file.")
    parser.add_argument("--iters", type=int, default=1000, help="Number of additional iterations.")
    args = parser.parse_args()

    print(f"Loading '{args.file}'...")
    with open(args.file, "rb") as f:
        data = pickle.load(f)

    # Reconstruct data
    params_load = jnp.array(data["final_params"])  # Start from the final params of previous run
    mask_load = jnp.array(data["mask"])
    state_load = data["optimizer_state"]  # This should be a valid Optax state (PyTree)
    
    # Constants (Must match 2nd_optim.py)
    wl = 1.064
    T = 70.0
    dk1 = mgoslt.calc_twm_delta_k(wl, wl, T)
    dk2 = mgoslt.calc_twm_delta_k(wl, wl/2, T)
    
    kappa_shg_val = 1.5e-5 / (2 / jnp.pi)
    b_init = jnp.array([jnp.sqrt(10.0), 0.0, 0.0], dtype=jnp.complex128)
    BLOCK_SIZE = 100
    
    # Reconstruct k1/k2 from mask size
    # We assume standard alternating signs logic was used
    n_active = len(mask_load) # Note: mask might have padding, but k1/k2 needs to match full shape
    # In 2nd_optim, k1/k2 are created based on MAX_N (padded size). 
    # 'mask_load' likely has size MAX_N (9000).
    pad_len = 0 # If mask is already padded
    
    # We can infer the pattern from the mask. 
    # However, k1/k2 generation logic in 2nd_optim was:
    # signs_active = np.tile([1.0, -1.0], n_active // 2 + 1)[:n_active] 
    # But wait, we don't know the exact n_active used to generate the padding if we only have the mask.
    # Actually, mask has 1s and 0s. Sum(mask) gives active length.
    n_real_active = int(jnp.sum(mask_load))
    total_len = len(mask_load)
    
    signs_active = np.tile([1.0, -1.0], n_real_active // 2 + 1)[:n_real_active]
    signs_padded = np.pad(signs_active, (0, total_len - n_real_active), mode='constant', constant_values=0.0)
    
    k1 = jnp.array(signs_padded * kappa_shg_val)
    k2 = jnp.array(signs_padded * 2 * kappa_shg_val)
    
    # Optimizer
    optimizer = optax.lbfgs(learning_rate=1.0, memory_size=20, linesearch=optax.scale_by_backtracking_linesearch(max_backtracking_steps=10))
    
    # Determine which device the loaded state is on. If it was saved as numpy (pickled), jax can ingest it.
    # But optax states might need to be explicitly cast if they are raw numpy arrays now.
    # JAX handles numpy arrays natively, so passing them to scan should be fine, 
    # provided the structure is exactly correctly matching the optimizer def.
    
    # -------------------------------------------------------------------------
    # Update Step Function
    # -------------------------------------------------------------------------
    def step_fn(carry, i):
        p, state = carry
        
        def loss_fn(params):
            w = 1.5 + jax.nn.softplus(params)
            L_curr = jnp.dot(w, mask_load)
            len_penalty = 100.0 * (jax.nn.relu(L_curr - 20000.0)**2)
            
            res = cwes2.simulate_magnus(w, k1, k2, dk1, dk2, b_init, BLOCK_SIZE)
            amp = jnp.abs(res[2]) 
            return -amp + len_penalty, (amp, L_curr)

        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p)
        g = g * mask_load
        
        u, new_s = optimizer.update(g, state, p, value=l, grad=g, value_fn=lambda x: loss_fn(x)[0])
        new_p = optax.apply_updates(p, u)
        return (new_p, new_s), (l, aux)

    print(f"Running continuation for {args.iters} iterations...")
    start_t = time.time()
    
    @scan_tqdm(args.iters, print_rate=10)
    def scan_loop_body(carry, i):
        return step_fn(carry, i)

    (final_p, final_s), (losses, (amps, lens)) = jax.lax.scan(  # pyright: ignore[reportGeneralTypeIssues]
        scan_loop_body,
        (params_load, state_load),
        jnp.arange(args.iters)
    )
    
    print(f"Done in {time.time() - start_t:.2f}s")
    
    final_amp = amps[-1]
    prev_amp = data["final_amp"]
    
    print(f"Previous Amp: {prev_amp:.5f}")
    print(f"New Final Amp: {final_amp:.5f}")
    
    # structures
    w_final = 1.5 + jax.nn.softplus(final_p)
    valid_indices = np.array(mask_load, dtype=bool)
    w_final_clean = np.array(w_final)[valid_indices]
    
    # Save
    out_name = f"continued_solution_{args.iters}.pkl"
    save_data = data.copy()
    save_data["final_params"] = np.array(final_p)
    save_data["final_amp"] = float(final_amp)
    save_data["final_structure"] = w_final_clean
    save_data["optimizer_state"] = final_s
    
    with open(out_name, "wb") as f:
        pickle.dump(save_data, f)
    
    print(f"Saved result to '{out_name}'")

if __name__ == "__main__":
    main()
