import pickle
import numpy as np


def main():
    path = "best_sol.pkl"
    print(f"Loading {path}...")
    with open(path, "rb") as f:
        data = pickle.load(f)

    w_final = np.array(data["final_structure"])

    # Calculate periods: w0+w1, w2+w3, w4+w5, ...
    # Ensure even length for pairing
    n_pairs = len(w_final) // 2
    w_pairs = w_final[: 2 * n_pairs].reshape(-1, 2)
    periods = w_pairs.sum(axis=1)

    min_period = np.min(periods)
    min_idx = np.argmin(periods)

    print(f"Total domains: {len(w_final)}")
    print(f"Number of periods (pairs): {len(periods)}")
    print(f"Minimum period: {min_period:.6f} µm (at index {min_idx * 2}, {min_idx * 2 + 1})")

    # Also check minimum single domain width just in case
    min_w = np.min(w_final)
    print(f"Minimum single domain width: {min_w:.6f} µm")


if __name__ == "__main__":
    main()
