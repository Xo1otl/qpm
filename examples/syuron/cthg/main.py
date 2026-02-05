import jax

# Enable x64 precision
jax.config.update("jax_enable_x64", val=True)

from structure import SimulationConfig, setup_structure
from compute import run_magnus, run_dop853
from viz import plot_results


def run_simulation_campaign(
    config: SimulationConfig,
    title: str,
    output_filename: str
) -> None:
    print(f"\n=== {title} ===")
    
    print("Setting up structure...")
    struct = setup_structure(config)
    print(f"Structure setup complete. Num domains: {len(struct.domain_widths)}")

    print(f"\nRunning Super Step (Magnus) on {title}...")
    magnus_res, time_magnus = run_magnus(struct)
    print(f"Super Step time: {time_magnus:.6f} s")

    print(f"\nRunning Reference (SciPy DOP853) on {title}...")
    dop853_res, time_dop853 = run_dop853(struct)
    print(f"SciPy DOP853 time: {time_dop853:.6f} s")

    print(f"\nPlotting results for {title}...")
    plot_results(magnus_res, dop853_res, time_magnus, time_dop853, filename=output_filename)


def main() -> None:
    # --- 1. Ideal Structure ---
    run_simulation_campaign(
        config=SimulationConfig(),
        title="Ideal Structure",
        output_filename="amp_trace_comparison_ideal.html"
    )

    # --- 2. Noisy Structure ---
    # Add 1% standard deviation noise relative to coherence length? 
    # approx lc ~ 7.5um. 0.05um noise?
    # User said "some noize". Let's try 0.1 um noise standard deviation.
    NOISE_STD = 0.1 
    run_simulation_campaign(
        config=SimulationConfig(domain_width_noise_std=NOISE_STD),
        title=f"Noisy Structure (std={NOISE_STD})",
        output_filename="amp_trace_comparison_noisy.html"
    )
    
    print("\nDone.")


if __name__ == "__main__":
    main()
