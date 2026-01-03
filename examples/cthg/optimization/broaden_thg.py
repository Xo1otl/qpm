from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
import plotly.graph_objects as go
from jax import Array, jit
from jax_tqdm.loop_pbar import loop_tqdm

from qpm import cwes, mgoslt

# --- Constants ---
FW_INITIAL_RATIO = 1 / 3


# --- Data Structures for Configuration and Parameters ---
@dataclass(frozen=True)
class Config:
    """Stores all hyperparameters for the optimization and simulation."""

    design_temp: float = 70.0
    target_length: float = 500.0
    max_iters: int = 100
    prng_seed: int = 42
    kappa_mag: float = 1.31e-4 / (2 / jnp.pi)
    # Full spectrum calculation range
    wl_start: float = 1.025
    wl_end: float = 1.035
    num_wl_points: int = 500
    # Target flat-top region for the loss function
    flat_top_wl_start: float = 1.031 - 0.0015
    flat_top_wl_end: float = 1.031 + 0.0015
    target_power: float = 0.002


@dataclass(frozen=True)
class SimulationParameters:
    """Stores fixed parameters required for the simulation physics."""

    kappa_array: Array
    b_initial: Array
    wls: Array
    delta_k1s: Array
    delta_k2s: Array
    flat_top_indices: tuple[Array, ...]


# --- Core JAX Functions ---
def make_broadening_loss_fn(sim_params: SimulationParameters, config: Config) -> Callable[[Array], Array]:
    """Creates a loss function that calculates the MSE between the simulated power
    in a specified spectral region and a target power.
    """
    compute_spectrum_vmap = jit(jax.vmap(cwes.simulate_twm, in_axes=(None, None, 0, 0, None)))

    @jit
    def loss_fn(domain_widths: Array) -> Array:
        # 1. Simulate the full power spectrum
        final_vectors = compute_spectrum_vmap(
            domain_widths,
            sim_params.kappa_array,
            sim_params.delta_k1s,
            sim_params.delta_k2s,
            sim_params.b_initial,
        )
        thw_powers = jnp.abs(final_vectors[:, 2]) ** 2

        # 2. Slice the spectrum to the target region
        flat_top_powers = thw_powers[sim_params.flat_top_indices]

        # 3. Calculate and return the Mean Squared Error loss
        return jnp.mean(jnp.square(flat_top_powers - config.target_power))

    return loss_fn


# --- Workflow Functions ---
def initialize_simulation(config: Config) -> tuple[Array, SimulationParameters]:
    """Generates initial domain widths and fixed simulation parameters."""
    print("1. Generating initial grating and simulation parameters...")

    # --- Grating Initialization ---
    center_wl = (config.flat_top_wl_start + config.flat_top_wl_end) / 2
    delta_k2_center = mgoslt.calc_twm_delta_k(center_wl, center_wl / 2, config.design_temp)

    domain_len_thg = jnp.pi / delta_k2_center
    num_domains = int(jnp.round(config.target_length / domain_len_thg))
    print(f"   - Base domain width from calculation: {domain_len_thg:.4f} μm")
    print(f"   - Total number of domains: {num_domains}")

    kappa_array = config.kappa_mag * jnp.power(-1.0, jnp.arange(num_domains))

    key = jax.random.PRNGKey(config.prng_seed)
    noise = jax.random.normal(key, shape=(num_domains,)) * (domain_len_thg * 0.02)
    initial_widths = jnp.full(num_domains, domain_len_thg) + noise

    # --- Simulation Parameter Preparation ---
    wls = jnp.linspace(config.wl_start, config.wl_end, config.num_wl_points)
    delta_k1s = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)
    delta_k2s = mgoslt.calc_twm_delta_k(wls, wls / 2, config.design_temp)
    flat_top_indices = jnp.where((wls >= config.flat_top_wl_start) & (wls <= config.flat_top_wl_end))
    b_initial = jnp.array(
        [
            jnp.sqrt(FW_INITIAL_RATIO),
            jnp.sqrt(FW_INITIAL_RATIO * 2),
            0.0,
        ],
        dtype=jnp.complex64,
    )

    sim_params = SimulationParameters(
        kappa_array=kappa_array,
        b_initial=b_initial,
        wls=wls,
        delta_k1s=delta_k1s,
        delta_k2s=delta_k2s,
        flat_top_indices=flat_top_indices,
    )
    return initial_widths, sim_params


def run_optimization(initial_widths: Array, sim_params: SimulationParameters, config: Config) -> tuple[Array, Array]:
    """Sets up and runs the L-BFGS optimization."""
    print("2. Setting up loss function and L-BFGS optimizer...")
    loss_fn = make_broadening_loss_fn(sim_params, config)
    solver = optax.lbfgs()

    @jit
    def run_full_optimization(initial_params: Array) -> tuple[Array, Array]:
        initial_loss_history = jnp.zeros(config.max_iters)

        @loop_tqdm(config.max_iters, desc="Optimizing")
        def step(i: int, state: tuple[Array, optax.OptState, Array]) -> tuple[optax.Params, optax.OptState, Array]:
            params, opt_state, loss_history = state
            value, grad = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=loss_fn)
            params = optax.apply_updates(params, updates)
            loss_history = loss_history.at[i].set(value)
            return params, opt_state, loss_history

        initial_opt_state = solver.init(initial_params)
        optimized_params, _, loss_history = jax.lax.fori_loop(
            0,
            config.max_iters,
            step,
            (initial_params, initial_opt_state, initial_loss_history),
        )
        return optimized_params, loss_history

    print("3. Running JIT-compiled optimization...")
    print(f"   - Initial Loss: {loss_fn(initial_widths):.4e}")

    optimized_widths, loss_history = run_full_optimization(initial_widths)
    optimized_widths.block_until_ready()

    print(f"   - Optimized Loss: {loss_fn(optimized_widths):.4e}")
    return optimized_widths, loss_history


# --- Visualization Functions ---
def calculate_spectrum(domain_widths: Array, sim_params: SimulationParameters) -> tuple[Array, Array]:
    """Helper function to compute the THW power spectrum for a given grating."""
    compute_spectrum_vmap = jit(jax.vmap(cwes.simulate_twm, in_axes=(None, None, 0, 0, None)))

    final_vectors = compute_spectrum_vmap(
        domain_widths,
        sim_params.kappa_array,
        sim_params.delta_k1s,
        sim_params.delta_k2s,
        sim_params.b_initial,
    )
    thw_powers = jnp.abs(final_vectors[:, 2]) ** 2
    return sim_params.wls, thw_powers


def plot_domain_widths(initial_widths: Array, optimized_widths: Array, periodic_widths: Array) -> None:
    """Plots the domain widths before and after optimization."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=initial_widths, mode="lines", name="Initial", line={"dash": "dash"}))
    fig.add_trace(go.Scatter(y=periodic_widths, mode="lines", name="Periodic", line={"dash": "dot"}))
    fig.add_trace(go.Scatter(y=optimized_widths, mode="lines", name="Optimized"))
    fig.update_layout(
        title_text="Domain Widths Comparison",
        xaxis_title="Domain Index",
        yaxis_title="Width (μm)",
        template="plotly_white",
    )
    fig.show()


def plot_thw_power_spectrum(
    initial_widths: Array,
    optimized_widths: Array,
    periodic_widths: Array,
    sim_params: SimulationParameters,
    config: Config,
) -> None:
    """Plots the THW power spectrum before and after optimization."""
    wls, initial_effs = calculate_spectrum(initial_widths, sim_params)
    _, optimized_effs = calculate_spectrum(optimized_widths, sim_params)
    _, periodic_effs = calculate_spectrum(periodic_widths, sim_params)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wls, y=initial_effs, mode="lines", name="Initial", line={"dash": "dash"}))
    fig.add_trace(go.Scatter(x=wls, y=periodic_effs, mode="lines", name="Periodic", line={"dash": "dot"}))
    fig.add_trace(go.Scatter(x=wls, y=optimized_effs, mode="lines", name="Optimized"))
    fig.add_hline(
        y=config.target_power,
        line_width=2,
        line_dash="dash",
        line_color="green",
        annotation_text="Target Power",
    )
    fig.add_vrect(
        x0=config.flat_top_wl_start,
        x1=config.flat_top_wl_end,
        fillcolor="red",
        opacity=0.1,
        line_width=0,
        annotation_text="Target Region",
        annotation_position="top left",
    )
    fig.update_layout(
        title_text="THG Power Spectrum Comparison",
        xaxis_title="Fundamental Wavelength (μm)",
        yaxis_title="THW Power",
        template="plotly_white",
    )
    fig.show()


def plot_cost_history(loss_history: Array) -> None:
    """Plots the optimization cost history."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_history, mode="lines", name="Loss"))
    fig.update_layout(
        title_text="Optimization Cost History",
        xaxis_title="Iteration",
        yaxis_title="Loss (MSE)",
        template="plotly_white",
    )
    fig.show()


def main() -> None:
    """Main function to run the optimization and visualization."""
    config = Config()

    # 1. Generate initial domain widths and fixed simulation parameters
    initial_widths, sim_params = initialize_simulation(config)

    # Create the ideal periodic grating for comparison
    center_wl = (config.flat_top_wl_start + config.flat_top_wl_end) / 2
    delta_k2_center = mgoslt.calc_twm_delta_k(center_wl, center_wl / 2, config.design_temp)
    domain_len_thg = jnp.pi / delta_k2_center
    num_domains = int(jnp.round(config.target_length / domain_len_thg))
    periodic_widths = jnp.full(num_domains, domain_len_thg)

    # 2. Run the optimization
    optimized_widths, loss_history = run_optimization(initial_widths, sim_params, config)

    # 3. Visualize optimization results
    print("4. Visualizing optimization results...")
    plot_domain_widths(initial_widths, optimized_widths, periodic_widths)
    plot_thw_power_spectrum(initial_widths, optimized_widths, periodic_widths, sim_params, config)
    plot_cost_history(loss_history)


if __name__ == "__main__":
    main()
