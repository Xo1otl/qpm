import pickle
import numpy as np
import jax.numpy as jnp
import jax
import qpm.cwes2 as cwes2
from maximize_amp_lbfgs import get_initial_structure
from analyze_checkpoint import SimulationConfig

# Enable x64 for precision
jax.config.update("jax_enable_x64", val=True)


def main():
    pkl_path = "optimized_best.pkl"
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 1. Best Structure
    widths_best = np.abs(np.array(data["params"]))
    n_best = len(widths_best)

    cfg = data.get("config", SimulationConfig())
    if isinstance(cfg, dict):
        cfg = SimulationConfig(**cfg)

    print(f"Configuration: Pin={cfg.input_power}")

    # Reconstruct Environment for Best
    widths_initial_arr, k_shg_init, k_sfg_init, (dk1, dk2, _, _) = get_initial_structure(cfg)

    # Initial Structure (Analytical Design)
    widths_initial = np.abs(widths_initial_arr)
    n_initial = len(widths_initial)

    # Construct kappa arrays for Best
    signs_best = jnp.ones(n_best)
    signs_best = signs_best.at[1::2].set(-1.0)
    k_shg_best = signs_best * cfg.kappa_shg
    k_sfg_best = signs_best * cfg.kappa_sfg

    # Construct b_initial vector for simulate_twm
    amp_fund = jnp.sqrt(cfg.input_power)
    b_init_vec = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)

    print("-" * 30)
    print("Simulating INITIAL Structure (TWM)...")

    b_final_init = cwes2.simulate_twm(jnp.array(widths_initial), k_shg_init, k_sfg_init, dk1, dk2, b_init_vec)

    amp_init = jnp.abs(b_final_init[2])
    inten_init = amp_init**2
    print(f"Initial Intensity: {inten_init:.6f}")

    print("-" * 30)
    print("Simulating BEST Structure (TWM)...")
    b_final_best = cwes2.simulate_twm(jnp.array(widths_best), k_shg_best, k_sfg_best, dk1, dk2, b_init_vec)
    amp_best = jnp.abs(b_final_best[2])
    inten_best = amp_best**2
    print(f"Best Intensity: {inten_best:.6f}")

    print("-" * 30)
    ratio = inten_init / inten_best
    print(f"Ratio (Initial / Best): {ratio:.6f}")
    print(f"Improvement (Best / Initial): {1.0 / ratio:.6f}x")
    print("-" * 30)


if __name__ == "__main__":
    main()
