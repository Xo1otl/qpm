import jax
import jax.numpy as jnp
import numpy as np
from qpm import cwes2, mgoslt

def verify_sub_optimal():
    print("Verifying Sub optimal Unconstrained Structure...")
    jax.config.update("jax_enable_x64", True)
    
    # Physics
    wl = 1.064
    T = 70.0
    dk1 = mgoslt.calc_twm_delta_k(wl, wl, T)
    dk2 = mgoslt.calc_twm_delta_k(wl, wl/2, T)
    lc1 = float(jnp.pi / jnp.abs(dk1))
    lc2 = float(jnp.pi / jnp.abs(dk2))
    
    print(f"Lc1: {lc1:.4f} um")
    print(f"Lc2: {lc2:.4f} um")
    
    kappa_shg_val = 1.5e-5 / (2 / np.pi)
    b_init = jnp.array([np.sqrt(10.0), 0.0, 0.0], dtype=jnp.complex128)
    BLOCK_SIZE = 300
    
    # Construction: "fixed-cycle Lc linking"
    # Section 1: SHG (L=10000)
    L_shg_target = 10000.0
    n_shg = int(np.round(L_shg_target / lc1))
    w_shg = [lc1] * n_shg
    print(f"SHG Section: {len(w_shg)} domains, Length ~ {np.sum(w_shg):.2f} um")
    
    # Section 2: SFG (L=7500) - Unconstrained!
    # Using Lc2 directly (approx 1.1 um)
    L_sfg_target = 7500.0
    n_sfg = int(np.round(L_sfg_target / lc2))
    w_sfg = [lc2] * n_sfg
    print(f"SFG Section: {len(w_sfg)} domains, Length ~ {np.sum(w_sfg):.2f} um")
    
    # Concatenate
    w_full = jnp.array(w_shg + w_sfg)
    N = len(w_full)
    L_total = jnp.sum(w_full)
    print(f"Total Structure: {N} domains, Length {L_total:.2f} um")
    
    # Signs (+ - + - ...)
    signs = jnp.tile(jnp.array([1.0, -1.0]), N // 2 + 1)[:N]
    k1 = signs * kappa_shg_val
    k2 = signs * 2 * kappa_shg_val
    
    # Simulate
    res = cwes2.simulate_magnus(w_full, k1, k2, dk1, dk2, b_init, BLOCK_SIZE)
    amp = float(jnp.abs(res[2]))
    
    print(f"\nFinal Amplitude |A3|: {amp:.5f}")
    
    if amp > 1.7:
        print("SUCCESS: Sub optimal structure exceeds 1.7 target!")
    else:
        print("FAILURE: Even sub optimal structure fails to reach 1.7.")

if __name__ == "__main__":
    verify_sub_optimal()
