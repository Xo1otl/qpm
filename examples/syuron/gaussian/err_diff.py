import matplotlib.pyplot as plt
import numpy as np

# Global scaling factor for styling
SCALE = 2


def plot_pwm_comparison_fixed():
    period = 8
    # User-requested duty set
    duties = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    n_domains = len(duties)
    total_samples = period * n_domains

    # Higher resolution for "Original" to show continuous widths
    res = 100
    t_fine = np.linspace(0, total_samples, total_samples * res)
    y_before = np.full_like(t_fine, -1.0)

    signal_after = []
    error = 0.0

    # Centers for labels
    centers_original = []
    centers_diffused = []
    w_afters = []

    # Original (ideal) signal: No quantization of width
    for i, d in enumerate(duties):
        ideal_width = d * period
        start_idx = i * period * res

        # Left-align the pulse within the period in the fine grid
        pulse_samples = int(ideal_width * res)
        pulse_start = start_idx
        y_before[pulse_start : pulse_start + pulse_samples] = 1.0

        # Center position for label (relative to pulse)
        centers_original.append(i * period + (ideal_width / 2 if ideal_width > 0 else period / 2))

        # Quantized signal (Error Diffusion)
        target_val = ideal_width + error
        # Round to nearest even number (scale of 2)
        w_after = int(np.round(target_val / 2) * 2)
        w_after = max(0, min(period, w_after))
        error = target_val - w_after
        w_afters.append(w_after)

        centers_diffused.append(i * period + (w_after / 2 if w_after > 0 else period / 2))

        def make_pulse(width):
            arr = np.full(period, -1.0)
            if width > 0:
                # Left-align
                arr[0:width] = 1.0
            return arr

        signal_after.append(make_pulse(w_after))

    y_after = np.concatenate(signal_after)
    # Append the last value to ensure the step plot reaches the end
    y_after_plot = np.append(y_after, y_after[-1])
    t_after_plot = np.arange(len(y_after_plot))

    # Styling based on SCALE
    fig_width = 10
    fig_height = 5
    font_size = 12 * SCALE
    line_width = 1.5 * SCALE
    text_size = 14 * SCALE

    plt.rcParams.update({"font.size": font_size})

    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    ticks = np.arange(0, total_samples + 1, period)

    # Plot Original (fine resolution)
    axes[0].plot(t_fine, y_before, color="black", linewidth=line_width)

    # Add width labels for Original
    for center, d in zip(centers_original, duties, strict=False):
        width = d * period
        axes[0].text(center, 1.3, f"{width:.1f}", ha="center", va="center", fontsize=text_size)

    # Plot Error Diffusion (step plot)
    axes[1].step(t_after_plot, y_after_plot, where="post", color="black", linewidth=line_width)

    # Add width labels for Error Diffusion
    for center, width in zip(centers_diffused, w_afters, strict=False):
        axes[1].text(center, 1.3, f"{width}", ha="center", va="center", fontsize=text_size, color="red")

    for ax in axes:
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 1])
        ax.set_xticks(ticks)
        ax.grid(axis="x", linestyle="--", color="black", alpha=0.3)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.axhline(y=-1.5, color="black", linewidth=1 * SCALE)

    axes[0].tick_params(labelbottom=False)

    plt.tight_layout()
    plt.savefig("err_diff.png")
    print("Saved err_diff.png")


if __name__ == "__main__":
    plot_pwm_comparison_fixed()
