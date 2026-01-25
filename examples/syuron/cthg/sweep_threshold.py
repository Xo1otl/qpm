import pickle
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Import functions from our analysis script
from analyze_checkpoint import (
    SimulationConfig,
    evaluate_structure,
    reconstruct_environment,
    simplify_structure,
)

# Enable x64 for precision (matches analyze_checkpoint.py)
jax.config.update("jax_enable_x64", val=True)


def main():
    # Base file (cleaned_structure.pkl is currently 0.5um threshold)
    # BUT, to be fair, we should start from a detailed structure if possible.
    # However, user's "original" was checkpoint_15000_cleaned.pkl which is gone.
    # cleaned_structure.pkl at 0.5um is already simplified.
    # To sweep "removing up to X", we should ideally use the backup of the 0.1um or 0.3um version
    # to see intermediate steps, OR just continue simplifying the 0.5um version further.
    # Since we are testing limits > 0.5um and we have a 0.5um structure, we can start there.
    # Or better: use the LEAST simplified available.
    # I have cleaned_structure_backup_0.1.pkl (from 0.3 step) and cleaned_structure_backup_0.3.pkl (from 0.5 step).
    # I'll try to load 'cleaned_structure_refined_0.1.pkl' if it exists, or 'cleaned_structure.pkl'.

    # Actually, `cleaned_structure.pkl` IS `cleaned_structure_refined_0.5.pkl`.
    # I should check if I have `cleaned_structure_refined_0.1.pkl` from previous step (I saved it but didn't explicitly check persistence, I used `cp`).
    # Let's just use `cleaned_structure.pkl` (0.5um) as baseline.
    # Any threshold < 0.5um will do nothing. We want to sweep > 0.5um.

    base_path = "cleaned_structure.pkl"
    print(f"Loading baseline from {base_path}...")

    with open(base_path, "rb") as f:
        data = pickle.load(f)

    params = data["params"]
    cfg = data.get("config", SimulationConfig())
    if isinstance(cfg, dict):
        cfg = SimulationConfig(**cfg)

    widths_base = np.abs(np.array(params))
    n_base = len(widths_base)

    # Reconstruct Env
    k_shg, k_sfg, dk1, dk2 = reconstruct_environment(cfg, n_base)
    amp_fund = jnp.sqrt(cfg.input_power)
    b_initial = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)

    # Baseline Eval
    amp_b, inten_b = evaluate_structure(jnp.array(widths_base), k_shg, k_sfg, dk1, dk2, b_initial)
    print(f"Baseline (0.5um) Intensity: {inten_b:.6f}")

    # Thresholds to sweep
    # Focus on 0.5 to 1.3
    thresholds = np.linspace(0.5, 1.3, 17)  # 0.5, 0.55, ..., 1.3

    results_n = []
    results_inten = []
    results_ratio = []

    print("\nStarting Sweep...")
    print(f"{'Threshold (um)':<15} | {'Domains':<10} | {'Intensity':<12} | {'Retention':<12}")
    print("-" * 60)

    for th in thresholds:
        # Simplify from BASE (0.5um)
        # Note: logic removes domains < th.
        widths_new = simplify_structure(widths_base, threshold=th)
        n_new = len(widths_new)

        # Eval
        k_shg_new, k_sfg_new, dk1_new, dk2_new = reconstruct_environment(cfg, n_new)
        amp_new, inten_new = evaluate_structure(jnp.array(widths_new), k_shg_new, k_sfg_new, dk1_new, dk2_new, b_initial)

        ratio = float(inten_new / inten_b)

        results_n.append(n_new)
        results_inten.append(float(inten_new))
        results_ratio.append(ratio)

        print(f"{th:<15.3f} | {n_new:<10} | {inten_new:<12.6f} | {ratio:<12.6f}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel("Threshold (um)")
    ax1.set_ylabel("Domain Count", color=color)
    ax1.plot(thresholds, results_n, color=color, marker="o", label="Domains")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:red"
    ax2.set_ylabel("Retention Ratio", color=color)  # we already handled the x-label with ax1
    ax2.plot(thresholds, results_ratio, color=color, marker="x", linestyle="--", label="Retention")
    ax2.tick_params(axis="y", labelcolor=color)

    # Add Lc Reference lines
    lc_apprx = 1.10  # SFG Lc
    plt.axvline(lc_apprx, color="green", linestyle=":", label="Lc SFG (~1.1um)")

    plt.title("Simplification Sweep: Threshold vs Efficiency")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("sweep_plot.png")
    print("\nSaved sweep_plot.png")

    # Save CSV
    with open("sweep_results.csv", "w") as f:
        f.write("Threshold,Domains,Intensity,Retention\n")
        for t, n, i, r in zip(thresholds, results_n, results_inten, results_ratio):
            f.write(f"{t},{n},{i},{r}\n")
    print("Saved sweep_results.csv")


if __name__ == "__main__":
    main()
