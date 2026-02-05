import jax
import jax.numpy as jnp
import optax
import numpy as np
from qpm import cwes2, mgoslt
import time
from jax_tqdm import scan_tqdm  # pyright: ignore[reportPrivateImportUsage]

def run_parallel_sweep_masked():
    BATCH_SIZE = 120
    
    print(f"Initializing Parallel Sweep with MASKING (Batch={BATCH_SIZE})...")
    jax.config.update("jax_enable_x64", True)
    
    wl = 1.064
    T = 70.0
    dk1 = mgoslt.calc_twm_delta_k(wl, wl, T)
    dk2 = mgoslt.calc_twm_delta_k(wl, wl/2, T)
    lc1 = float(jnp.pi / jnp.abs(dk1))
    
    kappa_shg_val = 1.5e-5 / (2 / jnp.pi)
    b_init = jnp.array([jnp.sqrt(10.0), 0.0, 0.0], dtype=jnp.complex128)
    BLOCK_SIZE = 300
    
    L_total = 17500.0
    MAX_N = 9000 
    
    period = 4.4257
    w1 = 1.51
    w2 = period - w1
    
    init_params_list = []
    mask_list = []
    k1_list = []
    k2_list = []
    
    splits_np = np.linspace(0.30, 0.70, BATCH_SIZE)
    
    for s in splits_np:
        L_shg_target = s * L_total
        n_shg = int(L_shg_target / lc1)
        w_curr = [lc1] * n_shg
        
        L_rem = L_total - np.sum(w_curr)
        n_periods = int(L_rem / period)
        for _ in range(n_periods):
            w_curr.append(w1)
            w_curr.append(w2)
        
        w_arr = np.array(w_curr)
        n_active = len(w_arr)
        
        if n_active > MAX_N:
            print(f"Error: MAX_N too small. Needed {n_active}")
            return

        pad_len = MAX_N - n_active
        w_padded = np.pad(w_arr, (0, pad_len), mode='constant', constant_values=lc1)
        
        p = np.log(np.exp(w_padded - 1.5) - 1.0)
        init_params_list.append(p)
        
        m = np.concatenate([np.ones(n_active), np.zeros(pad_len)])
        mask_list.append(m)
        
        signs_active = np.tile([1.0, -1.0], n_active // 2 + 1)[:n_active]
        signs_padded = np.pad(signs_active, (0, pad_len), mode='constant', constant_values=0.0)
        
        k1_list.append(signs_padded * kappa_shg_val)
        k2_list.append(signs_padded * 2 * kappa_shg_val)

    params_batch = jnp.array(init_params_list)
    mask_batch = jnp.array(mask_list)
    k1_batch = jnp.array(k1_list)
    k2_batch = jnp.array(k2_list)
    
    optimizer = optax.lbfgs(learning_rate=1.0, memory_size=20, linesearch=optax.scale_by_backtracking_linesearch(max_backtracking_steps=10))
    opt_state_batch = jax.vmap(optimizer.init)(params_batch)
    
    ITERATIONS = 3000

    # -------------------------------------------------------------------------
    # Update Step Function (Single Sample)
    # -------------------------------------------------------------------------
    def step_fn_single(p, state, k1_in, k2_in, mask_in):
        def loss_fn(params):
            w = 1.5 + jax.nn.softplus(params)
            L_curr = jnp.dot(w, mask_in)
            len_penalty = 100.0 * (jax.nn.relu(L_curr - 20000.0)**2)
            
            res = cwes2.simulate_magnus(w, k1_in, k2_in, dk1, dk2, b_init, BLOCK_SIZE)
            amp = jnp.abs(res[2]) 
            return -amp + len_penalty, (amp, L_curr)

        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p)
        g = g * mask_in
        
        u, new_s = optimizer.update(g, state, p, value=l, grad=g, value_fn=lambda x: loss_fn(x)[0])
        new_p = optax.apply_updates(p, u)
        return new_p, new_s, l, aux

    # -------------------------------------------------------------------------
    # Batch Step Function (vmap over batch)
    # -------------------------------------------------------------------------
    def step_fn_batch(carry, _):
        params_b, state_b = carry
        
        # vmap the update logic over the batch dimension
        new_params_b, new_state_b, losses, auxs = jax.vmap(step_fn_single)(
            params_b, state_b, k1_batch, k2_batch, mask_batch
        )
        
        return (new_params_b, new_state_b), (losses, auxs)

    # -------------------------------------------------------------------------
    # Scan with TQDM
    # -------------------------------------------------------------------------
    print("Running Parallel Sweep with Variable Topology...")
    start_t = time.time()

    @scan_tqdm(ITERATIONS, print_rate=10)
    def scan_loop_body(carry, i):
        return step_fn_batch(carry, i)

    (final_params_batch, final_opt_state_batch), (_, (batch_amps, batch_lens)) = jax.lax.scan(  # pyright: ignore[reportGeneralTypeIssues]
        scan_loop_body, 
        (params_batch, opt_state_batch), 
        jnp.arange(ITERATIONS)
    )

    print(f"Done in {time.time() - start_t:.2f}s")
    
    # Extract results from the final step of the history (or max over history)
    # Here we take the amps from the last iteration. 
    # If you need the best *ever* seen, you'd need to track it in 'carry', 
    # but L-BFGS is monotonic enough that last is usually best.
    
    # batch_amps shape: (ITERATIONS, BATCH) -> We want the last one
    final_amps = batch_amps[-1]
    final_lens = batch_lens[-1]
    
    best_idx = jnp.argmax(final_amps)
    best_amp = final_amps[best_idx]
    best_split = splits_np[best_idx]
    
    # --- Retrieve Initial Info for the Best Split ---
    best_init_p = params_batch[best_idx]
    best_mask = mask_batch[best_idx]
    best_k1 = k1_batch[best_idx]
    best_k2 = k2_batch[best_idx]

    # Calculate initial amplitude
    # We need to reconstruct the initial 'w' from 'best_init_p'
    w_init = 1.5 + jax.nn.softplus(best_init_p)
    # Run simulation for initial state
    res_init = cwes2.simulate_magnus(w_init, best_k1, best_k2, dk1, dk2, b_init, BLOCK_SIZE)
    init_amp = jnp.abs(res_init[2])

    print(f"Global Best Split: {best_split:.3f}")
    print(f"Initial Amp: {init_amp:.5f}")   # Added
    print(f"Final Amp:   {best_amp:.5f}")   # Modified label
    print(f"Length at Best: {final_lens[best_idx]:.2f}")
    
    # --- Prepare Structures for Output ---
    # Final Structure
    raw_p_final = final_params_batch[best_idx]
    w_final = 1.5 + jax.nn.softplus(raw_p_final)
    
    valid_indices = np.array(best_mask, dtype=bool)
    w_init_clean = np.array(w_init)[valid_indices]
    w_final_clean = np.array(w_final)[valid_indices]

    print("\n--- Initial Structure (Widths) ---")
    print(w_init_clean)
    print("\n--- Final Structure (Widths) ---")
    print(w_final_clean)

    # --- Save Solution & Gradients ---
    # Extract the optimizer state corresponding to the best index
    # opt_state usually is a tuple of (count, (grads, ...)) or similar depending on the optimizer.
    # We use jax.tree_map to pick the slice at best_idx for every leaf.
    best_opt_state = jax.tree.map(lambda x: x[best_idx], final_opt_state_batch)

    save_data = {
        "split": float(best_split),
        "initial_params": np.array(best_init_p),
        "final_params": np.array(raw_p_final),
        "mask": np.array(best_mask),
        "initial_structure": w_init_clean,
        "final_structure": w_final_clean,
        "initial_amp": float(init_amp),
        "final_amp": float(best_amp),
        "optimizer_state": best_opt_state  # Contains gradients/history
    }

    import pickle
    with open("best_solution_grad.pkl", "wb") as f:
        pickle.dump(save_data, f)
    print("\nSaved best solution and optimizer state to 'best_solution_grad.pkl'")


if __name__ == "__main__":
    run_parallel_sweep_masked()
