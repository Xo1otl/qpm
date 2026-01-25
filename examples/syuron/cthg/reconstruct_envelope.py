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
    windows = [10, 20, 30, 50, 100, 200]

    print("\n--- Envelope Smoothing Reconstruction ---")
    print(f"{'Window':<10} | {'Intensity':<12} | {'Retention':<10}")
    print("-" * 40)

    best_retention = 0.0
    best_w = None
    best_win = 0
    results_inten = []

    # Separation
    w_even = widths_orig[0::2]
    w_odd = widths_orig[1::2]

    for w_size in windows:
        # Smooth separately
        # Adjust window size if too large for even/odd arrays (half length)
        nx = len(w_even)
        eff_size = min(w_size, nx - 1)

        w_even_ma = uniform_filter1d(w_even, size=eff_size, mode="nearest")
        w_odd_ma = uniform_filter1d(w_odd, size=eff_size, mode="nearest")

        # Recombine
        widths_new = np.empty_like(widths_orig)
        widths_new[0::2] = w_even_ma
        widths_new[1::2] = w_odd_ma

        # Eval
        _, inten_ma = evaluate_structure(widths_new, k_shg, k_sfg, dk1, dk2, b_initial)

        retention = float(inten_ma / inten_base)
        results_inten.append(inten_ma)

        print(f"{w_size:<10} | {inten_ma:<12.6f} | {retention:<10.4f}")

        if retention > best_retention:
            best_retention = retention
            best_w = widths_new
            best_win = w_size

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 1. Structure Comparison
    ax1.plot(widths_orig, color="gray", alpha=0.5, linestyle="None", marker=".", markersize=1, label="Optimized (Original)")
    if best_w is not None:
        ax1.plot(
            best_w, color="green", linestyle="None", marker=".", markersize=1, label=f"Smoothed Envelope (W={best_win}, Ret={best_retention:.4f})"
        )

    ax1.set_title("Envelope Smoothing Reconstruction")
    ax1.set_xlabel("Domain Index")
    ax1.set_ylabel("Width (um)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Efficiency Sweep
    ax2.plot(windows, results_inten, marker="o", linestyle="-")
    ax2.axhline(inten_base, color="green", linestyle="--", label="Baseline")
    ax2.set_title("Efficiency vs Smoothing Window (Envelope)")
    ax2.set_xlabel("Window Size")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("envelope_reconstruction.png")
    print("\nSaved envelope_reconstruction.png")

    if best_retention > 0.99:
        print(f"\nSUCCESS: Envelope Smoothing with W={best_win} retains >99% efficiency!")
        # Save this structure if good
        out_pkl = "optimized_0.5_envelope.pkl"
        out_data = {"params": best_w, "config": cfg}
        with open(out_pkl, "wb") as f:
            pickle.dump(out_data, f)
        print(f"Saved smoothed structure to {out_pkl}")
    else:
        print(f"\nFAILURE: Best retention is only {best_retention * 100:.2f}%.")
        print("Conclusion: Even the envelope fluctuations contain critical phase information.")


if __name__ == "__main__":
    main()
