import pickle
import numpy as np
import matplotlib.pyplot as plt
from analyze_checkpoint import SimulationConfig


def main():
    pkl_path = "optimized_0.5_final.pkl"
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    widths = np.abs(np.array(data["params"]))
    n_domains = len(widths)

    # Test periodicities
    periods = [2, 3, 4, 6]

    fig, axes = plt.subplots(len(periods), 1, figsize=(15, 4 * len(periods)), sharex=True)
    if len(periods) == 1:
        axes = [axes]

    for i, N in enumerate(periods):
        ax = axes[i]

        # Color code by index % N
        colors = plt.cm.tab10(np.linspace(0, 1, N))

        for r in range(N):
            # Select indices where index % N == r
            indices = np.arange(r, n_domains, N)
            vals = widths[indices]

            ax.scatter(indices, vals, s=0.5, color=colors[r], label=f"Mod {r}", alpha=0.7)

        ax.set_title(f"Assumed Periodicity N={N}")
        ax.set_ylabel("Width (um)")
        ax.legend(markerscale=10, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.xlabel("Domain Index")
    plt.tight_layout()
    plt.savefig("multi_envelope_analysis.png")
    print("Saved multi_envelope_analysis.png")


if __name__ == "__main__":
    main()
