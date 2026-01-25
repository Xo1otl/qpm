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

    # Calculate Period and Duty Cycle
    # We group domains in pairs (assuming alternating +/-)
    # Period P[k] = w[2k] + w[2k+1]
    # Duty Cycle D[k] = w[2k] / P[k]

    # If the structure is odd number of domains, drop last one
    if n_domains % 2 != 0:
        w_pairs = widths[:-1]
    else:
        w_pairs = widths

    w_even = w_pairs[0::2]
    w_odd = w_pairs[1::2]

    periods = w_even + w_odd
    duty_cycles = w_even / periods

    # Return Map (Phase Space)
    # Plot w[i+1] vs w[i]
    w_curr = widths[:-1]
    w_next = widths[1:]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    # 1. Local Period vs Index
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(np.arange(len(periods)), periods, s=1, color="purple", alpha=0.5)
    ax1.set_title("Local Period Evolution (P = w_i + w_{i+1})")
    ax1.set_xlabel("Period Index")
    ax1.set_ylabel("Period (um)")
    ax1.grid(True, alpha=0.3)

    # 2. Duty Cycle vs Index
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(np.arange(len(duty_cycles)), duty_cycles, s=1, color="orange", alpha=0.5)
    ax2.set_title("Duty Cycle Evolution (D = w_i / P)")
    ax2.set_xlabel("Period Index")
    ax2.set_ylabel("Duty Cycle")
    ax2.grid(True, alpha=0.3)

    # 3. Return Map (Phase Space)
    ax3 = fig.add_subplot(gs[1, :])
    # Color by index to show time evolution
    sc = ax3.scatter(w_curr, w_next, c=np.arange(len(w_curr)), cmap="viridis", s=1, alpha=0.5)
    ax3.set_title("Return Map (Phase Space Trajectory): w_{i+1} vs w_i")
    ax3.set_xlabel("Width[i] (um)")
    ax3.set_ylabel("Width[i+1] (um)")
    ax3.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax3, label="Domain Index")

    plt.tight_layout()
    plt.savefig("chirp_analysis.png")
    print("Saved chirp_analysis.png")


if __name__ == "__main__":
    main()
