import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.ndimage import uniform_filter1d
from analyze_checkpoint import SimulationConfig, reconstruct_environment, evaluate_structure

# Enable x64 for precision
import jax

jax.config.update("jax_enable_x64", val=True)


def main():
    pkl_path = "optimized_0.5_final.pkl"
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    widths_orig = np.abs(np.array(data["params"]))
    n_orig = len(widths_orig)
    cfg = data.get("config", SimulationConfig())
    if isinstance(cfg, dict):
        cfg = SimulationConfig(**cfg)

    # Baseline
    k_shg, k_sfg, dk1, dk2 = reconstruct_environment(cfg, n_orig)
    amp_fund = jnp.sqrt(cfg.input_power)
    b_initial = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)

    _, inten_base = evaluate_structure(widths_orig, k_shg, k_sfg, dk1, dk2, b_initial)
    print(f"Baseline Intensity: {inten_base:.6f}")

    # Windows to test
    # Note: Window of 20 roughly covers ~2-3 periods (if period ~8 index).
    # Window of 100 covers ~12 periods.
    windows = [10, 20, 30, 50, 100, 200]

    print("\n--- Moving Average Reconstruction ---")
    print(f"{'Window':<10} | {'Intensity':<12} | {'Retention':<10}")
    print("-" * 40)

    best_retention = 0.0
    best_w = None
    best_win = 0

    results_inten = []

    for w_size in windows:
        # Compute Moving Average (MA)
        # Using uniform_filter1d with 'nearest' to handle edges reasonable well
        widths_ma = uniform_filter1d(widths_orig, size=w_size, mode="nearest")

        # Eval
        # We perform exact simulation on this "smoothed" structure.
        # Physics: If we smooth widths, we remove fast phase corrections.
        # If those corrections were critical, efficiency will drop.
        # If they were noise, efficiency might hold up.
        _, inten_ma = evaluate_structure(widths_ma, k_shg, k_sfg, dk1, dk2, b_initial)

        retention = float(inten_ma / inten_base)
        results_inten.append(inten_ma)

        print(f"{w_size:<10} | {inten_ma:<12.6f} | {retention:<10.4f}")

        if retention > best_retention:
            best_retention = retention
            best_w = widths_ma
            best_win = w_size

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 1. Structure Comparison
    ax1.plot(widths_orig, color="gray", alpha=0.4, label="Optimized (Original)")
    ax1.plot(best_w, color="red", linewidth=2, label=f"Best MA (W={best_win}, Ret={best_retention:.4f})")
    # Also plot a heavily smoothed one for context if best is small
    if best_win < 100:
        w_heavy = uniform_filter1d(widths_orig, size=100, mode="nearest")
        ax1.plot(w_heavy, color="blue", linestyle="--", label="MA (W=100)")

    ax1.set_title("Structure Reconstruction from Moving Average")
    ax1.set_xlabel("Domain Index")
    ax1.set_ylabel("Width (um)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Efficiency Sweep
    ax2.plot(windows, results_inten, marker="o", linestyle="-")
    ax2.axhline(inten_base, color="green", linestyle="--", label="Baseline")
    ax2.set_title("Efficiency vs Smoothing Window")
    ax2.set_xlabel("Window Size")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ma_reconstruction.png")
    print("\nSaved ma_reconstruction.png")

    # Interpretation
    drop_percent = (1.0 - best_retention) * 100
    if best_retention > 0.99:
        print(f"\nSUCCESS: Smoothing with W={best_win} retains >99% efficiency!")
    else:
        print(f"\nFAILURE: Even mild smoothing W={best_win} drops efficiency by {drop_percent:.2f}%.")
        print("Conclusion: The high-frequency 'jitter' IS CRITICAL for phase matching.")


if __name__ == "__main__":
    main()
