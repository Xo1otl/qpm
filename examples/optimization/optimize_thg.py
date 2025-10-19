from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax  # pyright: ignore[reportMissingTypeStubs]
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
from jax import Array, jit
from jax_tqdm.loop_pbar import loop_tqdm

from qpm import cwes, mgoslt


# --- Regularization Strategies ---
def tv_regularization(domain_widths: Array, lambda_val: float) -> Array:
    """Total Variation (TV) regularization penalizes the absolute difference between adjacent parameter values."""
    return lambda_val * jnp.sum(jnp.abs(jnp.diff(domain_widths)))


REGULARIZATION_FNS: dict[str, Callable[[Array, float], Array]] = {
    "tv": tv_regularization,
}


# --- Data Structures for Configuration and Parameters ---
@dataclass(frozen=True)
class Config:
    design_temp: float = 70.0
    design_wl: float = 1.031
    target_length: float = 2000.0
    max_iters: int = 500
    prng_seed: int = 42
    kappa_mag: float = 1.31e-5 / (2 / jnp.pi)
    kappa_target_length: float = 2000.0
    wl_start: float = 1.025
    wl_end: float = 1.035
    num_points: int = 500
    # Regularization parameters
    regularization: str | None = "tv"
    lambda_val: float = 1e-4


@dataclass(frozen=True)
class SimulationParameters:
    delta_k1: Array
    delta_k2: Array
    kappa_array: Array
    b_initial: Array


# --- Core JAX Functions ---
def make_loss_fn(
    sim_params: SimulationParameters,
    config: Config,  # noqa: ARG001
    regularization_fn: Callable[[Array, float], Array] | None = None,
    lambda_val: float = 0.0,
) -> Callable[[Array], Array]:
    """Creates a loss function that returns the negative SHW power with optional regularization."""

    @jit
    def loss_fn(domain_widths: Array) -> Array:
        superlattice = jnp.stack([domain_widths, sim_params.kappa_array], axis=1)
        b_final = cwes.simulate_twm(superlattice, sim_params.delta_k1, sim_params.delta_k2, sim_params.b_initial)
        shw_power = jnp.abs(b_final[1]) ** 2

        # Base loss is negative SHW power (to maximize it)
        loss = -shw_power

        # Add regularization if provided
        if regularization_fn:
            loss += regularization_fn(domain_widths, lambda_val)

        return loss

    return loss_fn


# --- Workflow Functions ---
def _initialize_grating(config: Config) -> tuple[Array, Array, Array, Array]:
    """Generates initial grating parameters based on the configuration."""
    print("1. Generating initial grating parameters...")
    delta_k1 = mgoslt.calc_twm_delta_k(config.design_wl, config.design_wl, config.design_temp)
    delta_k2 = mgoslt.calc_twm_delta_k(config.design_wl, config.design_wl / 2, config.design_temp)

    domain_len_shg = jnp.pi / delta_k1
    num_domains = int(jnp.round(config.target_length / domain_len_shg))
    print(f"   - Base domain width from calculation: {domain_len_shg:.4f} μm")
    print(f"   - Number of domains: {num_domains}")

    kappa_array = config.kappa_mag * jnp.power(-1.0, jnp.arange(num_domains))

    initial_widths = jnp.full(num_domains, domain_len_shg)
    key = jax.random.PRNGKey(config.prng_seed)
    noise = jax.random.normal(key, shape=(num_domains,)) * (domain_len_shg * 0.02)
    initial_widths = initial_widths + noise

    return delta_k1, delta_k2, kappa_array, initial_widths


def _run_optimization(
    initial_widths: Array,
    sim_params: SimulationParameters,
    config: Config,
) -> Array:
    """Sets up and runs the L-BFGS optimization."""
    print("2. Setting up loss function and L-BFGS optimizer...")
    regularization_fn = REGULARIZATION_FNS.get(config.regularization) if config.regularization else None
    loss_fn = make_loss_fn(sim_params, config, regularization_fn, config.lambda_val)
    solver = optax.lbfgs()

    @jit
    def run_full_optimization(initial_params: Array) -> Array:
        @loop_tqdm(config.max_iters, desc="Optimizing")
        def step(_: int, state: tuple[Array, optax.OptState]) -> tuple[optax.Params, optax.OptState]:
            params, opt_state = state
            value, grad = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=loss_fn)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        initial_opt_state = solver.init(initial_params)
        optimized_params, _ = jax.lax.fori_loop(0, config.max_iters, step, (initial_params, initial_opt_state))
        return optimized_params

    print("3. Running JIT-compiled optimization...")
    print(f"   - Initial SHW Power: {-loss_fn(initial_widths):.4f}")

    optimized_widths = run_full_optimization(initial_widths)
    optimized_widths.block_until_ready()

    print(f"   - Optimized SHW Power: {-loss_fn(optimized_widths):.4f}")
    return optimized_widths


# --- Visualization Functions ---
def _calculate_spectrum(domain_widths: Array, sim_params: SimulationParameters, config: Config) -> tuple[Array, Array]:
    """Helper function to compute the SHW Power spectrum for a given grating."""
    wls = jnp.linspace(config.wl_start, config.wl_end, config.num_points)
    delta_k1s = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)
    delta_k2s = mgoslt.calc_twm_delta_k(wls, wls / 2, config.design_temp)

    compute_spectrum_vmap = jit(jax.vmap(cwes.simulate_twm, in_axes=(None, 0, 0, None)))

    superlattice = jnp.stack([domain_widths, sim_params.kappa_array], axis=1)
    final_vectors = compute_spectrum_vmap(superlattice, delta_k1s, delta_k2s, sim_params.b_initial)
    shw_powers = jnp.abs(final_vectors[:, 1]) ** 2 * 1.07 / 2.84 * 100
    return wls, shw_powers


def _plot_domain_widths(initial_widths: Array, optimized_widths: Array) -> None:
    """Plots the domain widths before and after optimization."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=initial_widths, mode="lines", name="Initial", line={"dash": "dash"}))
    fig.add_trace(go.Scatter(y=optimized_widths, mode="lines", name="Optimized"))
    fig.update_layout(
        title_text="Domain Widths Before and After Optimization",
        xaxis_title="Domain Index",
        yaxis_title="Width (μm)",
        template="plotly_white",
    )
    fig.show()


def _plot_shw_power_spectrum(
    initial_widths: Array,
    optimized_widths: Array,
    sim_params: SimulationParameters,
    config: Config,
) -> None:
    """Plots the SHG Power spectrum before and after optimization."""
    wls, initial_effs = _calculate_spectrum(initial_widths, sim_params, config)
    _, optimized_effs = _calculate_spectrum(optimized_widths, sim_params, config)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wls, y=initial_effs, mode="lines", name="Initial", line={"dash": "dash"}))
    fig.add_trace(go.Scatter(x=wls, y=optimized_effs, mode="lines", name="Optimized"))
    fig.add_vline(
        x=config.design_wl,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Design λ {config.design_wl}μm",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title_text="SHW Power Spectrum Comparison",
        xaxis_title="Fundamental Wavelength (μm)",
        yaxis_title="SHW Power",
        template="plotly_white",
    )
    fig.show()


def main() -> None:
    """Main script execution."""
    config = Config()

    # 1. Generate initial physical parameters and domain widths
    delta_k1, delta_k2, kappa_array, initial_widths = _initialize_grating(config)

    # 2. Create simulation parameters
    sim_params = SimulationParameters(
        delta_k1=delta_k1,
        delta_k2=delta_k2,
        kappa_array=kappa_array,
        b_initial=jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex64),
    )

    # 3. Run the optimization with the clean parameter objects
    optimized_widths = _run_optimization(initial_widths, sim_params, config)

    # 4. Visualize the results
    print("4. Visualizing optimization results...")
    _plot_domain_widths(initial_widths, optimized_widths)
    _plot_shw_power_spectrum(initial_widths, optimized_widths, sim_params, config)


main()
