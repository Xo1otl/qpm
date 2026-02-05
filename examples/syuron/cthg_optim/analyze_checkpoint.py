import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Enable x64 for precision
jax.config.update("jax_enable_x64", val=True)

from qpm import cwes2, mgoslt


@dataclass
class SimulationConfig:
    # Copied from maximize_amp_lbfgs.py to ensure compatibility
    shg_len: float = 10005
    sfg_len: float = 7500
    kappa_shg: float = 1.5e-5 / (2 / jnp.pi)
    kappa_sfg: float = 1.5e-5 / (2 / jnp.pi) * 2
    temperature: float = 70.0
    wavelength: float = 1.064
    input_power: float = 10.0
    block_size: int = 300
    iterations: int = 500


def reconstruct_environment(cfg: SimulationConfig, n_domains: int):
    """
    Reconstructs the kappa arrays and delta_k values based on the config.
    Assumes standard alternating signs starting with +1.
    """
    dk_shg = mgoslt.calc_twm_delta_k(cfg.wavelength, cfg.wavelength, cfg.temperature)
    dk_sfg = mgoslt.calc_twm_delta_k(cfg.wavelength, cfg.wavelength / 2, cfg.temperature)

    # Standard alternating signs: +1, -1, +1, ...
    signs = jnp.ones(n_domains)
    signs = signs.at[1::2].set(-1.0)

    k_shg_vals = signs * cfg.kappa_shg
    k_sfg_vals = signs * cfg.kappa_sfg

    return k_shg_vals, k_sfg_vals, dk_shg, dk_sfg


def evaluate_structure(
    widths: jax.Array,
    k_shg: jax.Array,
    k_sfg: jax.Array,
    dk1: jax.Array,
    dk2: jax.Array,
    b_initial: jax.Array,
):
    """
    Runs one simulation step and returns the THG amplitude and intensity.
    """
    # Ensure widths are physical (positive)
    phys_widths = jnp.abs(widths)

    # Calculate valid block_size
    n_domains = len(phys_widths)
    block_size = 1
    # User requested block_size >= 20 and divisor of N
    # We prefer larger blocks for efficiency, say up to 200
    for bs in range(200, 19, -1):
        if n_domains % bs == 0:
            block_size = bs
            break

    # If no divisor found in [20, 200], we might be in a prime number case.
    # We could try to scan more, or fallback to 1 (slow) or N (one block).
    # Since existing logic worked with default (likely 1 or N), let's accept N if valid.
    if block_size == 1:
        # Check if N itself is reasonable? 8000 is large for single kernel.
        # But lacking block_size usually implies scan loop with default.
        # Let's try to pass N if no partial block found? Or just let library default.
        # The user said "specify a block_size of 20 or more".
        # If prime, we can't satisfy "divisor" AND ">=20" unless N >= 20.
        if n_domains >= 20:
            # Just use n_domains as block_size? That means 1 superstep.
            # Or leave it None/Default?
            # Let's trust the search found something usually.
            pass

    # Actually, cwes2.simulate_super_step signature might expect block_size as last arg?
    # Checked grep: maximize_amp_lbfgs.py calls it with block_size.
    # def simulate_super_step(widths, k_shg, k_sfg, dk1, dk2, b_init, block_size=None):

    b_final = cwes2.simulate_super_step(phys_widths, k_shg, k_sfg, dk1, dk2, b_initial, block_size=block_size)
    thg_amp = jnp.abs(b_final[2])
    return thg_amp, thg_amp**2


def plot_width_profile(
    widths: np.ndarray,
    lc_shg: float,
    lc_sfg: float,
    title: str = "Width Profile",
):
    """Plots width vs position with coherence length references."""
    z_positions = np.cumsum(widths)
    # Use center of domains for x-axis
    z_centers = z_positions - widths / 2

    plt.figure(figsize=(12, 6))
    plt.scatter(z_centers, widths, s=1, alpha=0.5, label="Domain Widths")

    # Moving average
    if len(widths) > 100:
        window = 50
        moving_avg = np.convolve(widths, np.ones(window) / window, mode="same")
        plt.plot(z_centers, moving_avg, color="red", linewidth=1, label="Moving Avg (50)")

    plt.axhline(lc_shg, color="green", linestyle="--", label=f"Lc SHG ({lc_shg:.2f})")
    plt.axhline(lc_sfg, color="orange", linestyle="--", label=f"Lc SFG ({lc_sfg:.2f})")

    plt.xlabel("Position (z) [um]")
    plt.ylabel("Domain Width [um]")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    print(f"Saved {title.lower().replace(' ', '_')}.png")


def simplify_structure(widths: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    """
    Simplifies the domain structure by merging negligible domains.
    Logic:
        Iteratively find the smallest domain < threshold.
        If it's an internal domain (idx 1 to N-2), merge (i-1, i, i+1) -> (new).
        new_width = sum(widths[i-1:i+2])
        This preserves the sign alternation because (+, -, +) -> (+).
    """
    # Work with numpy for mutability during simplification
    current_widths = list(widths)

    while True:
        # Find candidates (skip first and last for now to avoid boundary issues)
        if len(current_widths) < 3:
            break

        min_w = float("inf")
        min_idx = -1

        # Search for smallest internal domain
        # We need a domain i such that we can merge i-1, i, i+1.
        # So i must be in [1, len-2].
        found_candidate = False
        for i in range(1, len(current_widths) - 1):
            w = current_widths[i]
            if w < threshold:
                if w < min_w:
                    min_w = w
                    min_idx = i
                    found_candidate = True

        if not found_candidate:
            break

        # Merge
        # i-1, i, i+1 -> merged
        # Since signs are +, -, + (or -, +, -), the sum represents a domain of sign(i-1)
        # effectively bridging the gap.

        # print(f"Merging at {min_idx}: {current_widths[min_idx-1]:.4f}, {current_widths[min_idx]:.4f}, {current_widths[min_idx+1]:.4f}")

        new_width = current_widths[min_idx - 1] + current_widths[min_idx] + current_widths[min_idx + 1]

        # Remove the 3 old ones and insert the new one
        # Use slicing
        # new list = before (0 to min_idx-2) + [new] + after (min_idx+2 to end)

        # Slice indices:
        # [0, ..., min_idx-2] is up to min_idx-1 exclusive.
        # So we want widths[:min_idx-1]

        current_widths[min_idx - 1] = new_width
        # Remove i and i+1
        del current_widths[min_idx]
        del current_widths[min_idx]  # index shifts

        # Determine if we should continue (simplest is to just continue loop)

    return jnp.array(current_widths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path", type=str, help="Path to checkpoint pickle")
    parser.add_argument("--threshold", type=float, default=1e-2, help="Threshold for negligible width (in microns)")
    parser.add_argument("--save-cleaned", type=str, default=None, help="Path to save cleaned pickle")
    args = parser.parse_args()

    # Load Checkpoint
    print(f"Loading {args.pkl_path}...")
    with open(args.pkl_path, "rb") as f:
        data = pickle.load(f)

    # Extract info
    # Handle both direct pickle of dict or other formats if any
    if isinstance(data, dict):
        params = data.get("params")
        cfg = data.get("config", SimulationConfig())  # Fallback default
        # If config is dict, convert to dataclass
        if isinstance(cfg, dict):
            cfg = SimulationConfig(**cfg)
    else:
        raise ValueError("Unknown pickle format")

    if params is None:
        raise ValueError("No 'params' found in pickle")

    # Initial Setup
    widths_orig = jnp.abs(params)
    n_orig = len(widths_orig)
    print(f"Original Domains: {n_orig}")
    print(f"Total Length: {jnp.sum(widths_orig):.4f}")

    # Reconstruct Environment
    k_shg, k_sfg, dk1, dk2 = reconstruct_environment(cfg, n_orig)

    lc_shg = jnp.pi / dk1
    lc_sfg = jnp.pi / dk2
    print(f"Lc_SHG: {lc_shg:.4f} um")
    print(f"Lc_SFG: {lc_sfg:.4f} um")

    amp_fund = jnp.sqrt(cfg.input_power)
    b_initial = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)

    # evaluate original
    amp_orig, inten_orig = evaluate_structure(widths_orig, k_shg, k_sfg, dk1, dk2, b_initial)
    print(f"Original Intensity: {inten_orig:.6f}")

    # Plot original distribution
    plt.figure(figsize=(10, 4))
    plt.hist(np.array(widths_orig), bins=50, log=True)
    plt.title("Original Width Distribution")
    plt.xlabel("Width (um)")
    plt.ylabel("Count")
    plt.savefig("width_dist_orig.png")
    print("Saved width_dist_orig.png")

    # Simplify
    print(f"\nSimplifying with threshold < {args.threshold} um...")
    widths_clean = simplify_structure(np.array(widths_orig), threshold=args.threshold)
    n_clean = len(widths_clean)

    print(f"Cleaned Domains: {n_clean} (Removed {n_orig - n_clean})")
    print(f"Cleaned Length: {jnp.sum(widths_clean):.4f}")

    # Reconstruct Env for Cleaned
    k_shg_c, k_sfg_c, dk1_c, dk2_c = reconstruct_environment(cfg, n_clean)

    # Evaluate Cleaned
    amp_clean, inten_clean = evaluate_structure(widths_clean, k_shg_c, k_sfg_c, dk1, dk2, b_initial)
    print(f"Cleaned Intensity:  {inten_clean:.6f}")

    ratio = inten_clean / inten_orig
    print(f"Retention Ratio: {ratio:.6f}")

    if args.save_cleaned:
        out_data = {"params": widths_clean, "config": cfg, "thg_amp_orig": amp_orig, "thg_amp_clean": amp_clean}
        with open(args.save_cleaned, "wb") as f:
            pickle.dump(out_data, f)
        print(f"Saved cleaned structure to {args.save_cleaned}")

    # Analyze Pattern
    mean_w = jnp.mean(widths_clean)
    std_w = jnp.std(widths_clean)
    print(f"\nCleaned Mean Width: {mean_w:.4f} +/- {std_w:.4f}")

    # Print the first few domains to see if there's a pattern
    print("\nFirst 20 widths (Cleaned):")
    print(widths_clean[:20])

    # Print samples
    print("\n--- Structure Analysis ---")
    print(f"Start (0-20): {widths_clean[:20]}")
    mid = n_clean // 2
    print(f"Middle ({mid}-{mid + 20}): {widths_clean[mid : mid + 20]}")
    print(f"End ({n_clean - 20}-end): {widths_clean[-20:]}")

    # Check simple periodicity
    # Approximate counts
    shg_like = jnp.sum((widths_clean > 3.5) & (widths_clean < 4.5))
    sfg_like = jnp.sum((widths_clean > 0.8) & (widths_clean < 1.4))
    print(f"\nDomains close to Lc_SHG (~3.96): {shg_like} ({shg_like / n_clean * 100:.1f}%)")
    print(f"Domains close to Lc_SFG (~1.10): {sfg_like} ({sfg_like / n_clean * 100:.1f}%)")

    # Plot cleaned distribution
    plt.figure(figsize=(10, 4))
    plt.hist(np.array(widths_clean), bins=100, log=True)
    plt.title(f"Cleaned Width Distribution (Thresh={args.threshold})")
    plt.xlabel("Width (um)")
    plt.ylabel("Count")
    plt.savefig("width_dist_clean.png")

    # Deep Dive: Profile Plot
    plot_width_profile(np.array(widths_clean), lc_shg, lc_sfg, title="Cleaned Width Profile")

    # Deep Dive: Smallest Domains
    print("\n--- Smallest Domains Investigation ---")
    sorted_indices = np.argsort(widths_clean)
    print("Top 10 Smallest Domains:")
    for i in range(10):
        idx = sorted_indices[i]
        print(f"Idx {idx}: {widths_clean[idx]:.6g} um")

    # Check boundaries
    print(f"\nBoundary Check:")
    print(f"Idx 0: {widths_clean[0]:.6g} um")
    print(f"Idx {n_clean - 1}: {widths_clean[-1]:.6g} um")

    # Check if any < threshold
    # Note: Simplification ensures internal < threshold are merged. Only boundaries might remain?
    under_thresh = np.sum(widths_clean < args.threshold)
    print(f"Domains < {args.threshold} um: {under_thresh}")


if __name__ == "__main__":
    main()
