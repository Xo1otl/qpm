import argparse
import jax

jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
from bayes_opt import BayesianOptimization
from qpm import cwes2
import ppln_opt
import plot_comparison  # for build_3seg_structure logic if needed, or re-use ppln_opt


# Force CPU
def calculate_fw95m(x: jax.Array, y: jax.Array) -> float:
    """
    Calculate the full width at 95% of the maximum intensity.
    Returns width in same units as x.
    """
    max_val = jnp.max(y)
    threshold = 0.95 * max_val
    valid_indices = jnp.where(y >= threshold)[0]
    if valid_indices.size == 0:
        return 0.0
    x_start = x[valid_indices[0]]
    x_end = x[valid_indices[-1]]
    return float(x_end - x_start)


def run_bayesian_optimization(n_iter: int, init_points: int):
    # 1. Setup base config (mostly for Reference Calculation and fixed params)
    base_cfg = ppln_opt.get_default_config()

    # Pre-calculate fixed structures and reference intensity
    print("Pre-calculating reference structures...")
    initial_widths, kappa_vals = ppln_opt.build_paper_structure(base_cfg)
    max_intensity = ppln_opt.get_reference_intensity(base_cfg, initial_widths)
    print(f"Reference Max Intensity: {max_intensity:.4e}")

    # 2. Define Objective Function
    def objective_function(target_min, target_max, target_intensity):
        # Constraint: min < max
        if target_min >= target_max:
            return -1.0  # Penalize invalid range

        # Create config for this iteration
        # Note: target_flat_range uses min and max.
        # "target_flat_range uses its min and max as independent variables, each moving within a range of ±0.001."
        # The prompt implies we optimize the BOUNDS of the flat range.

        current_cfg = ppln_opt.SimConfig(
            grating_period=base_cfg.grating_period,
            kappa=base_cfg.kappa,
            target_flat_range=(target_min, target_max),
            target_normalized_intensity=target_intensity,
            default_iterations=300,  # "The optimizer will run for 300 loops each time."
            b_initial=base_cfg.b_initial,
            delta_k2=base_cfg.delta_k2,
        )

        # Run Inner Optimization (L-BFGS)
        # We start from the same initial simulation structure each time to be fair
        final_params, _, _ = ppln_opt.run_optimization(
            current_cfg, initial_widths, kappa_vals, max_intensity, iterations=current_cfg.default_iterations
        )

        # Simulate Resulting Spectrum
        final_widths_abs = jnp.abs(final_params)

        dk_center = 2.0 * jnp.pi / current_cfg.grating_period
        dk_scan = jnp.linspace(dk_center * 0.999, dk_center * 1.001, 1000)

        batch_sim = jax.vmap(cwes2.simulate_shg_npda, in_axes=(None, None, 0, None))
        amps = batch_sim(final_widths_abs, kappa_vals, dk_scan, base_cfg.b_initial[0])
        intensity = jnp.abs(amps) ** 2

        # Calculate Metrics
        width_95 = calculate_fw95m(dk_scan, intensity)

        # Objective: Maximize (95% width) * (target_intensity)
        # Note: Prompt says "maximizing (95% width) * (target_intensity)"
        # Assuming target_intensity is the parameter we set, not the achieved one.
        # But usually we care about achieved. However, the optimizer tries to match target.
        # So using the parameter is a good proxy if optimization works.
        score = width_95 * target_intensity

        return float(score)

    # 3. Setup Bayesian Optimizer
    # Limits:
    # target_normalized_intensity: 0.2 to 0.3
    # target_flat_range: center around default 1.9446 and 1.9459?
    # "target_flat_range uses its min and max as independent variables, each moving within a range of ±0.001."
    # Original default range: (1.9446, 1.9459)
    # So min is in [1.9436, 1.9456]
    # max is in [1.9449, 1.9469]

    pbounds = {"target_min": (1.9446 - 0.001, 1.9446 + 0.001), "target_max": (1.9459 - 0.001, 1.9459 + 0.001), "target_intensity": (0.2, 0.3)}

    optimizer = BayesianOptimization(f=objective_function, pbounds=pbounds, random_state=1, verbose=2)

    print("Starting Bayesian Optimization...")
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

    print("\nBest Result:")
    print(optimizer.max)
    return optimizer.max


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=100, help="Number of BO iterations")
    parser.add_argument("--init", type=int, default=10, help="Number of BO initial points")
    args = parser.parse_args()

    run_bayesian_optimization(args.iter, args.init)
