import matplotlib.pyplot as plt


def plot_bayesian_results(
    dk_scan: jax.Array,
    ref_int: jax.Array,
    opt_int: jax.Array,
    loss_hist: jax.Array,
    detected_range: tuple[float, float],
    plot_target: jax.Array,
) -> None:
    """
    Generates and saves optimization result plots with DETECTED range overlay.
    """

    # Calculate target intensity from target amplitude profile for plotting
    target_profile_int = plot_target**2

    # Spectrum Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dk_scan, ref_int, "--", color="gray", alpha=0.5, label="Uniform Ref")
    # Plot detected range as a shaded region or span
    # We plot this BEHIND the main line
    if detected_range[1] > detected_range[0]:
        plt.axvspan(detected_range[0], detected_range[1], color="orange", alpha=0.2, label="Detected 95% Flat-Top")

    plt.plot(dk_scan, opt_int, "-", color="#2E86AB", linewidth=2, label="Optimized")

    # Plot SG Target (just for reference of shape, not bounds)
    plt.plot(dk_scan, target_profile_int, "r:", linewidth=2, label="SG Target Shape")

    plt.legend()
    plt.title("Bayesian Optimization Result")
    plt.xlabel(r"$\Delta k$")
    plt.ylabel("Normalized Intensity")
    plt.savefig("bayesian_best_spectrum.png")
    plt.close()

    # Loss History Plot
    plt.figure(figsize=(10, 6))
    plt.plot(jnp.arange(len(loss_hist)), loss_hist, "-", color="#E63946")
    plt.yscale("log")
    plt.title("Optimization Loss History")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig("bayesian_best_loss.png")
    plt.close()
