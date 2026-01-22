import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_phasor_vectors():
    # Parameters provided by user feedback:
    # >7 vectors on a single circle.
    # At least 2 vectors rotated > pi (pointing down).

    # We define vectors v_n = exp(i * n * d_phi)
    # The sum is E_N = sum(v_n).
    # With Delta k mismatch, vectors rotate.
    # For a perfect "arc", we just plot the sequence of vectors.

    # Let's use N=9 vectors to cover range 0 to > pi.
    # Step size: pi/6 (30 degrees).
    # Phases: 0, 30, 60, 90, 120, 150, 180 (pi), 210, 240.
    # Vectors: 0..6 are "up" (imag >= 0) or forward.
    # Vector 7 (210 deg) and 8 (240 deg) have imaginary part < 0 (pointing down relative to horizontal, or "back").

    d_phi_deg = 25  # Slightly less than 30 to fit more
    d_phi = np.deg2rad(d_phi_deg)

    # We want "2 pointing down" (phase > pi).
    # If we stop at phase = 225 deg?
    # Let's simply generate a sequence until we pass a certain total phase.

    vectors_mismatch = []
    current_angle = 0.0
    # Add vectors until we have enough
    # We want the LAST ones to be pointing "down" (angle > pi/2 ? or > pi?)
    # "rotated more than pi" means cumulative phase > pi.
    # A vector exp(i * phi) points "down" if pi < phi < 2pi.

    # Let's generate 10 vectors.
    N = 12
    for i in range(N):
        # angle for the i-th slab
        angle = i * d_phi
        vectors_mismatch.append(np.exp(1j * angle))

    # --- Visualization ---
    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Common plotting helper
    def plot_chain(ax, vectors, title, color, show_circle=True, qpm_flip=False):
        ax.set_title(title, fontsize=24)
        ax.set_aspect("equal")

        # Starting point
        pos = 0 + 0j

        # For circle fitting (Mismatch case only) - skipped implementation details for brevity
        if show_circle:
            pass

        points_complex = [0 + 0j]
        curr = 0 + 0j
        for v in vectors:
            curr += v
            points_complex.append(curr)

        # Plot Arrows
        for i, v in enumerate(vectors):
            start = points_complex[i]

            ax.arrow(start.real, start.imag, v.real, v.imag, head_width=0.1, head_length=0.15, fc=color, ec=color, length_includes_head=True)

        # Axis limits
        points = np.array([[z.real, z.imag] for z in points_complex])
        all_x = points[:, 0]
        all_y = points[:, 1]
        pad = 1.0

        # Ensure consistent scale if possible, but auto-scale is fine for shape comparison
        # QPM and PM grow large, Mismatch stays small.
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        ax.axhline(0, color="k", alpha=0.1)
        ax.axvline(0, color="k", alpha=0.1)

        # Remove axes (lines, numbers, ticks) for cleanliness
        ax.axis("off")

    # 1. Phase Matching (PM) - Straight line
    vectors_pm = [1.0 + 0j] * N
    plot_chain(axes[0], vectors_pm, "Phase Matching", color="tab:green")

    # 2. Phase Mismatch (Mismatch) - Circle
    plot_chain(axes[1], vectors_mismatch, "Phase Mismatch", color="tab:orange")

    # 3. Quasi-Phase Matching (QPM) - Flipped
    vectors_qpm = []
    for i in range(N):
        angle = i * d_phi
        norm_angle = angle % (2 * np.pi)
        v = np.exp(1j * angle)
        if norm_angle >= np.pi:  # In the destructive zone
            v = -v  # Flip!
        vectors_qpm.append(v)

    plot_chain(axes[2], vectors_qpm, "QPM", color="tab:blue", qpm_flip=True)

    plt.tight_layout()
    plt.savefig("qpm_vector_chain.png", dpi=300)
    print("Saved qpm_vector_chain.png")


if __name__ == "__main__":
    plot_phasor_vectors()
