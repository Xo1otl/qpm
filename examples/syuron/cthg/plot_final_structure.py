import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d
from analyze_checkpoint import SimulationConfig, reconstruct_environment


def main():
    pkl_path = "optimized_best.pkl"
    output_png = "optimized_best_width_profile.png"

    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    widths = np.abs(np.array(data["params"]))
    n_domains = len(widths)
    cfg = data.get("config", SimulationConfig())
    if isinstance(cfg, dict):
        cfg = SimulationConfig(**cfg)

    # Calculate Physics Constants
    # We need n_domains to reconstruct environment correctly
    k_shg, k_sfg, dk1, dk2 = reconstruct_environment(cfg, n_domains)

    # Lc = pi / dk
    # dk1, dk2 are scalars (or 0-d arrays) returned by reconstruct_environment
    lc_shg = np.pi / np.abs(dk1)
    lc_sfg = np.pi / np.abs(dk2)

    print(f"Lc SHG: {lc_shg:.4f} um")
    print(f"Lc SFG: {lc_sfg:.4f} um")

    # Moving Average
    window_size = 50
    widths_ma = uniform_filter1d(widths, size=window_size, mode="nearest")

    print(f"Plotting {n_domains} domains to {output_png}...")

    # Set large font sizes globally
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.titlesize": 24,
        }
    )

    # Use seaborn style for nicer look
    sns.set_style("whitegrid")

    # Create plot
    plt.figure(figsize=(14, 8))

    # Use scatter with small markers to avoid "solid block" look
    plt.scatter(np.arange(n_domains), widths, s=2, color="navy", alpha=0.3, label="Domain Width")

    # Plot Moving Average
    plt.plot(widths_ma, color="cyan", linewidth=2, label=f"Moving Avg (W={window_size})")

    # Plot Lc Lines
    plt.axhline(lc_shg, color="red", linestyle="--", linewidth=2, label=f"Lc SHG ({lc_shg:.2f} µm)")
    plt.axhline(lc_sfg, color="orange", linestyle="--", linewidth=2, label=f"Lc SFG ({lc_sfg:.2f} µm)")

    total_length = np.sum(widths)
    plt.title(f"Optimized Domain Width Profile (N={n_domains}, L={total_length:.2f} µm)")
    plt.xlabel("Domain Index")
    plt.ylabel("Width (µm)")

    # Force legend to show all
    plt.legend(loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Saved plot to {output_png}")


if __name__ == "__main__":
    main()
