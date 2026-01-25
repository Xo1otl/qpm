import pickle
import numpy as np
import jax.numpy as jnp
import argparse
from analyze_checkpoint import SimulationConfig, simplify_structure, reconstruct_environment, evaluate_structure


def final_cleanup(pkl_path, output_path, threshold=0.5):
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    params = data["params"]
    cfg = data.get("config", SimulationConfig())
    if isinstance(cfg, dict):
        cfg = SimulationConfig(**cfg)

    widths = np.abs(np.array(params))
    n_orig = len(widths)
    print(f"Original Domains: {n_orig}")

    # 1. Check for domains < threshold
    small_domains = np.sum(widths < threshold)
    print(f"Domains < {threshold} um: {small_domains}")

    # Simplify (remove < threshold)
    # Note: simplify_structure usually merges negligible domains.
    # If optimization created new small domains, this cleans them.
    widths_clean = simplify_structure(widths, threshold=threshold)
    print(f"After Threshold Cleanup: {len(widths_clean)} (Removed {n_orig - len(widths_clean)})")

    # 2. Handle the "single domain with period 3.x at the very end"
    # User said "removing this shouldn't impact efficiency".
    # Let's inspect the last few domains
    print("Last 10 domains:", widths_clean[-10:])

    # Look for a domain ~3.x near the end.
    # Note: period 3.x -> width ~1.5 ? Or width ~3.x?
    # User said "period of around 3.x". SHG Lc is ~3.96 (period ~8). SFG Lc is ~1.1 (period ~2.2).
    # Width ~3.3 was seen.
    # Let's find any domain > 2.5 near the end (last 20).

    candidates = []
    for i in range(len(widths_clean) - 20, len(widths_clean)):
        if widths_clean[i] > 2.0:
            candidates.append((i, widths_clean[i]))

    print("Large domains near end (>2.0):", candidates)

    if candidates:
        # User said "single domain". Assume the most prominent one.
        # If there's a 3.3 value (seen before), that's likely it.
        target_idx, target_val = candidates[-1]  # Pick the last one
        print(f"Removing anomaly at index {target_idx} (width {target_val:.4f})")

        # Cleanup logic for single internal domain:
        # Merge it with neighbors? Or just delete?
        # User said "removing this".
        # If we delete, we flip signs effectively.
        # Safer: Merge i-1, i, i+1?
        # If w[i] is the anomaly 3.3.
        # Merging with neighbors (likely ~1.1) -> 1.1 + 3.3 + 1.1 = 5.5. That's huge.
        # If it's truly an anomaly to be REMOVED, maybe it means set its width to 0 (merge neighbors)?
        # 1.1 (neg) + 3.3 (pos) + 1.1 (neg).
        # If we remove 3.3, we have 1.1 (neg) + 1.1 (neg) -> 2.2 (neg).
        # This seems physically plausible: 3.3 was an extra half-period inserted.
        # Removing it leaves the background periodicity.
        # So I will merge (i-1) + (i+1) and DELETE i.
        # Wait, simplify_structure merges i-1, i, i+1.
        # Here we want to Delete i?
        # Logic:
        # w_new = w[i-1] + w[i+1] (ignoring w[i])?
        # No, physically, light travels through w[i-1], then w[i], then w[i+1].
        # If we assume w[i] (3.3) was "wrongly inserted", maybe we should just join w[i-1] and w[i+1].
        # Since signs alternate + - + -,
        # w[i-1] (+) -> w[i] (-) -> w[i+1] (+)
        # If we remove w[i], we have w[i-1] (+) -> w[i+1] (+).
        # These merge into one large (+) domain.
        # Result width = w[i-1] + w[i+1].
        # If w[i] was 3.3 and w[neighbors] were ~1.1.
        # Result w = 1.1 + 1.1 = 2.2. (Period matches ~2.2 for SFG!).
        # This makes perfect sense. The anomaly 3.3 breaks the 1.1 pattern.
        # Removing it (merging prior and next) restores the 1.1 (actually 2.2 half-period? no 2.2 period -> 1.1 width).
        # Wait. 1.1 is Lc (half period). Period is 2.2.
        # w (+) = 1.1. w (-) = 1.1.
        # If we have 1.1 -> 3.3 -> 1.1.
        # That's w (+) -> w (-) -> w (+).
        # If 3.3 is (-) ... that's 3x Lc.
        # If we remove it ... 1.1 (+) -> 1.1 (+) -> Merged 2.2 (+).
        # That's 2x Lc. Effectively skipping a beat?
        # Or maybe the 3.3 is actually 3x 1.1, i.e., 3 consecutive domains merged?
        # If so, it's valid.
        # But user says "redundant... removing shouldn't impact".
        # I will execute the merge: w_new = w[i-1] + w[i+1]. Delete w[i].
        # And since w[i-1] and w[i+1] become adjacent and same sign, they merge.

        target_idx = int(target_idx)
        w_prev = float(widths_clean[target_idx - 1])
        w_next = float(widths_clean[target_idx + 1]) if target_idx + 1 < len(widths_clean) else 0.0

        new_width = w_prev + w_next

        # Convert to list for mutability and deletion
        w_list = list(widths_clean)
        w_list[target_idx - 1] = new_width

        # Delete i and i+1 (step i+1 is now merged into i-1)
        # Note: We need to handle indices carefully.
        # Original: ... i-1, i, i+1 ...
        # i-1 becomes sum. i and i+1 are removed?
        # Yes, because i+1 is merged into i-1.

        # Using list for deletion
        w_list = list(widths_clean)
        if target_idx + 1 < len(w_list):
            print(f"Merging {w_prev:.4f} + (skip {target_val:.4f}) + {w_next:.4f} -> {new_width:.4f}")
            del w_list[target_idx]
            del w_list[target_idx]  # index of i+1 is now i
        else:
            # Boundary case: Last domain is anomaly?
            print("Anomaly is last domain. Just removing it.")
            del w_list[target_idx]

        widths_clean = np.array(w_list)

    print(f"Final Count: {len(widths_clean)}")

    # Evaluate
    k_shg, k_sfg, dk1, dk2 = reconstruct_environment(cfg, len(widths_clean))
    amp_fund = jnp.sqrt(cfg.input_power)
    b_initial = jnp.array([amp_fund, 0.0, 0.0], dtype=jnp.complex128)
    amp, inten = evaluate_structure(widths_clean, k_shg, k_sfg, dk1, dk2, b_initial)
    print(f"Final Intensity: {inten:.6f}")

    # Save
    out_data = {"params": widths_clean, "config": cfg}
    with open(output_path, "wb") as f:
        pickle.dump(out_data, f)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    final_cleanup("optimized_0.5.pkl", "optimized_0.5_final.pkl", threshold=0.5)
