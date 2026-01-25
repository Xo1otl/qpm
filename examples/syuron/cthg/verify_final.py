import pickle
import numpy as np
import jax.numpy as jnp
import jax
from analyze_checkpoint import SimulationConfig, reconstruct_environment
import qpm.cwes2 as cwes2
import sys

# Enable x64 for precision
jax.config.update("jax_enable_x64", val=True)


def main():
    pkl_path = "optimized_best.pkl"
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    widths = np.abs(np.array(data["params"]))
    n_domains = len(widths)
    cfg = data.get("config", SimulationConfig())
    if isinstance(cfg, dict):
        cfg = SimulationConfig(**cfg)

    print(f"Configuration: L_SHG={cfg.shg_len}, L_SFG={cfg.sfg_len}, Pin={cfg.input_power}")
    print(f"Domains: {n_domains}")

    # Simulation
    k_shg, k_sfg, dk1, dk2 = reconstruct_environment(cfg, n_domains)
    amp_fund = jnp.sqrt(cfg.input_power)
    b_initial = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)

    # Calculate correct block_size
    block_size = 1
    # Try to find a good divisor
    for bs in range(200, 19, -1):
        if n_domains % bs == 0:
            block_size = bs
            break

    if len(sys.argv) > 1:
        block_size = int(sys.argv[1])
        print(f"OVERRIDE Block Size: {block_size}")
    else:
        print(f"Using Block Size: {block_size}")

    b_final = cwes2.simulate_super_step(jnp.array(widths), k_shg, k_sfg, dk1, dk2, b_initial, block_size=block_size)

    amp_thg = jnp.abs(b_final[2])
    inten_thg = amp_thg**2

    print("-" * 30)
    print(f"FINAL INTENSITY: {inten_thg:.6f}")
    print(f"FINAL EFFICIENCY: {inten_thg / cfg.input_power * 100:.2f}% (Relative)")
    print("-" * 30)


if __name__ == "__main__":
    main()
