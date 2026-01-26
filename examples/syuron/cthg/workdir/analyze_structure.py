import pickle
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("Analyzing best_polished_long.pkl...")
    try:
        with open("best_polished_long.pkl", "rb") as f:
            data = pickle.load(f)
            widths = data["widths"]
            amp = data["amp"]
    except FileNotFoundError:
        print("File not found.")
        return

    widths = np.abs(widths)
    n = len(widths)

    print(f"Amplitude: {amp}")
    print(f"Number of domains: {n}")
    print(f"Total Length: {np.sum(widths):.4f} um")

    # Check violations
    min_w = 1.5
    violations = widths[widths < min_w]
    n_violations = len(violations)

    print(f"Min Width: {np.min(widths):.4f} um")
    print(f"Max Width: {np.max(widths):.4f} um")
    print(f"Number of violations (<{min_w}um): {n_violations}")

    if n_violations > 0:
        print("Violations present! The penalty was likely insufficient.")
        print(f"Smallest widths: {np.sort(violations)[:10]}")
    else:
        print("Structure respects constraints.")

    # Histogram
    plt.figure()
    plt.hist(widths, bins=50)
    plt.title("Domain Width Distribution")
    plt.xlabel("Width (um)")
    plt.ylabel("Count")
    plt.axvline(x=1.5, color="r", linestyle="--")
    plt.savefig("structure_hist_long.png")

    # Plot first 100 widths
    plt.figure()
    plt.plot(widths[:100], "o-")
    plt.title("First 100 Domains")
    plt.ylabel("Width (um)")
    plt.savefig("structure_zoom_long.png")


if __name__ == "__main__":
    main()
