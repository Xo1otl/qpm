import pickle
import numpy as np
import jax.numpy as jnp
import jax
import random
from analyze_checkpoint import SimulationConfig, reconstruct_environment, evaluate_structure

# Enable x64 for precision to avoid JAX type errors
jax.config.update("jax_enable_x64", val=True)


def merge_mutation(widths):
    """
    Merges 3 consecutive domains into 1.
    Physically equivalent to flipping the sign of the middle domain (i)
    so it matches neighbors (i-1) and (i+1).
    Reduces domain count by 2.
    """
    n = len(widths)
    if n < 3:
        return None

    # Select random index i (middle one)
    # Must have i-1 and i+1
    idx = random.randint(1, n - 2)

    new_w = list(widths)
    # Signs are alternating.
    # If we merge i-1, i, i+1, we create one large domain with sign of i-1.
    merged_val = new_w[idx - 1] + new_w[idx] + new_w[idx + 1]

    # Replace i-1 with merged value
    new_w[idx - 1] = merged_val
    # Delete i and i+1
    del new_w[idx]
    del new_w[idx]  # Index i+1 is now at i

    return np.array(new_w)


def swap_mutation(widths):
    """
    Swaps two adjacent domains widths.
    """
    n = len(widths)
    if n < 2:
        return None

    idx = random.randint(0, n - 2)

    new_w = widths.copy()
    new_w[idx], new_w[idx + 1] = new_w[idx + 1], new_w[idx]

    return new_w


def main():
    pkl_path = "optimized_0.5_final.pkl"
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    current_widths = np.abs(np.array(data["params"]))
    # Ensure raw numpy array for manipulation
    current_widths = np.array(current_widths, dtype=np.float64)

    cfg = data.get("config", SimulationConfig())
    if isinstance(cfg, dict):
        cfg = SimulationConfig(**cfg)

    print(f"Configuration: Input Power={cfg.input_power}")

    # Constant physics inputs
    amp_fund = jnp.sqrt(cfg.input_power)
    # Force complex128
    b_initial = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)

    def get_fitness(w):
        # N changes, so we must reconstruct environment (k, signs) every time
        k_shg, k_sfg, dk1, dk2 = reconstruct_environment(cfg, len(w))
        # Ensure w is jne array for JAX, probably handled by evaluate_structure logic or JAX fits
        _, inten = evaluate_structure(w, k_shg, k_sfg, dk1, dk2, b_initial)
        return float(inten)

    print("Evaluating baseline...")
    current_score = get_fitness(current_widths)
    print(f"Baseline Intensity: {current_score:.6f} (Domains: {len(current_widths)})")

    best_widths = current_widths
    best_score = current_score

    iterations = 50  # Steps allowed to move
    candidates_per_step = 50  # Attempts per step to find an improving move

    print(f"Starting Hill Climber ({iterations} steps, {candidates_per_step} candidates/step)...")
    print(f"{'Step':<5} | {'Best Score':<12} | {'Domains':<8} | {'Note'}")
    print("-" * 50)

    found_improvement = False

    for step in range(iterations):
        candidates = []
        for _ in range(candidates_per_step):
            r = random.random()
            if r < 0.5:
                # Merge (Sign flip)
                cand_w = merge_mutation(best_widths)
            else:
                # Swap
                cand_w = swap_mutation(best_widths)

            if cand_w is not None:
                candidates.append(cand_w)

        # Evaluate candidates
        step_best_w = None
        step_best_s = -1.0

        # Simple scan
        for w in candidates:
            try:
                s = get_fitness(w)
                if s > step_best_s:
                    step_best_s = s
                    step_best_w = w
            except Exception:
                continue

        # Check if we improved over GLOBAL best
        if step_best_s > best_score:
            print(f"{step:<5} | {step_best_s:<12.6f} | {len(step_best_w):<8} | IMPROVEMENT!")
            best_score = step_best_s
            best_widths = step_best_w
            found_improvement = True
        else:
            # If no improvement globally, we might still want to accept local best to explore?
            # Or strict hill climber?
            # User said "completely stuck". Let's stick to Strict Hill Climber for now.
            pass

        if step % 10 == 0:
            print(f"{step:<5} | {best_score:<12.6f} | {len(best_widths):<8} | ...")

    print("-" * 50)
    print(f"Final Best: {best_score:.6f}")

    if found_improvement:
        print("Saving improved topology...")
        out_pkl = "optimized_topology.pkl"
        out_data = {"params": best_widths, "config": cfg}
        with open(out_pkl, "wb") as f:
            pickle.dump(out_data, f)
        print(f"Saved to {out_pkl}")
    else:
        print("No topological improvement found.")
        print("The Local Minimum is extremely deep.")


if __name__ == "__main__":
    main()
