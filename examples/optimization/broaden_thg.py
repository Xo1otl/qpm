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
    target_length: float = 1000.0
    max_iters: int = 200
    prng_seed: int = 42
    kappa_mag: float = 1.31e-4 / (2 / jnp.pi)
    kappa_target_length: float = 2000.0
    # Full spectrum calculation range
    wl_start: float = 1.025
    wl_end: float = 1.035
    num_points: int = 500
    # Target flat-top region for the loss function
    flat_top_wl_start: float = 1.0305
    flat_top_wl_end: float = 1.0315
    regularization: str | None = "tv"
    lambda_val: float = 1e-3


@dataclass(frozen=True)
class SimulationParameters:
    kappa_array: Array
    b_initial: Array


# --- Core JAX Functions ---
def make_simulation_fns(
    sim_params: SimulationParameters,
    config: Config,
    regularization_fn: Callable[[Array, float], Array] | None = None,
    lambda_val: float = 0.0,
) -> tuple[Callable[[Array], Array], Callable[[Array], tuple[Array, Array]]]:
    """Factory to create loss and spectrum functions with pre-computed values captured in closures."""
    # Pre-compute all wavelength-dependent arrays once
    wls = jnp.linspace(config.wl_start, config.wl_end, config.num_points)
    delta_k1s = mgoslt.calc_twm_delta_k(wls, wls, config.design_temp)
    delta_k2s = mgoslt.calc_twm_delta_k(wls, wls / 2, config.design_temp)
    flat_top_indices = jnp.where((wls >= config.flat_top_wl_start) & (wls <= config.flat_top_wl_end))

    # JIT-compiled vmap for efficient spectrum calculation
    compute_spectrum_vmap = jit(jax.vmap(cwes.simulate_twm, in_axes=(None, 0, 0, None)))

    @jit
    def spectrum_fn(domain_widths: Array) -> tuple[Array, Array]:
        """Calculates the full THW power spectrum for a given grating."""
        superlattice = jnp.stack([domain_widths, sim_params.kappa_array], axis=1)
        final_vectors = compute_spectrum_vmap(superlattice, delta_k1s, delta_k2s, sim_params.b_initial)
        thw_powers = jnp.abs(final_vectors[:, 2]) ** 2
        return wls, thw_powers

    @jit
    def loss_fn(half_widths: Array) -> Array:
        """Computes the logarithmic loss over the specified flat-top region."""
        # 1. Enforce symmetry
        domain_widths = jnp.concatenate([half_widths, half_widths[::-1]])

        # 2. Calculate the full spectrum
        _, thw_powers = spectrum_fn(domain_widths)

        # 3. Slice the spectrum and apply logarithmic loss
        flat_top_powers = thw_powers[flat_top_indices]
        epsilon = 1e-9
        log_loss = -jnp.mean(jnp.log(flat_top_powers + epsilon))

        # 4. Add regularization
        loss = log_loss
        if regularization_fn:
            loss += regularization_fn(half_widths, lambda_val)
        return loss

    return loss_fn, spectrum_fn


# --- Workflow Functions ---
def _initialize_grating(config: Config) -> tuple[Array, Array]:
    """Generates initial grating parameters for symmetric optimization."""
    print("1. Generating initial grating parameters (symmetric)...")
    # Use the center of the flat-top region for initial domain length calculation
    center_wl = (config.flat_top_wl_start + config.flat_top_wl_end) / 2
    delta_k2 = mgoslt.calc_twm_delta_k(center_wl, center_wl / 2, config.design_temp)

    domain_len_thg = jnp.pi / delta_k2
    num_domains = int(jnp.round(config.target_length / domain_len_thg))
    if num_domains % 2 != 0:
        num_domains += 1  # Ensure even number of domains for symmetry
    num_half_domains = num_domains // 2
    print(f"   - Base domain width from calculation: {domain_len_thg:.4f} μm")
    print(f"   - Total number of domains: {num_domains} (optimizing {num_half_domains})")

    kappa_array = config.kappa_mag * jnp.power(-1.0, jnp.arange(num_domains))

    initial_half_widths = jnp.full(num_half_domains, domain_len_thg)
    key = jax.random.PRNGKey(config.prng_seed)
    noise = jax.random.normal(key, shape=(num_half_domains,)) * (domain_len_thg * 0.02)
    initial_half_widths = initial_half_widths + noise

    return kappa_array, initial_half_widths


def _run_optimization(
    initial_half_widths: Array,
    loss_fn: Callable[[Array], Array],
    config: Config,
) -> Array:
    """Sets up and runs the L-BFGS optimization."""
    print("2. Setting up L-BFGS optimizer...")
    solver = optax.lbfgs()
    opt_state = solver.init(initial_half_widths)

    @jit
    def step(params: optax.Params, opt_state: optax.OptState) -> tuple[optax.Params, optax.OptState]:
        value, grad = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=loss_fn)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    @loop_tqdm(config.max_iters, desc="Optimizing")
    def run_loop(i: int, state: tuple[Array, optax.OptState]) -> tuple[Array, optax.OptState]:
        params, opt_state = state
        return step(params, opt_state)

    print("3. Running JIT-compiled optimization...")
    print(f"   - Initial Loss: {loss_fn(initial_half_widths):.4f}")

    optimized_half_widths, _ = jax.lax.fori_loop(0, config.max_iters, run_loop, (initial_half_widths, opt_state))
    optimized_half_widths.block_until_ready()

    print(f"   - Optimized Loss: {loss_fn(optimized_half_widths):.4f}")
    return optimized_half_widths


# --- Visualization Functions ---


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


def _plot_thw_power_spectrum(
    initial_widths: Array,
    optimized_widths: Array,
    spectrum_fn: Callable[[Array], tuple[Array, Array]],
    config: Config,
) -> None:
    """Plots the THW power spectrum before and after optimization using the provided spectrum function."""
    wls, initial_effs = spectrum_fn(initial_widths)
    _, optimized_effs = spectrum_fn(optimized_widths)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wls, y=initial_effs, mode="lines", name="Initial", line={"dash": "dash"}))
    fig.add_trace(go.Scatter(x=wls, y=optimized_effs, mode="lines", name="Optimized"))
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


def main() -> None:
    """Main script execution."""
    config = Config()

    # 1. Generate initial physical parameters for the symmetric grating
    kappa_array, initial_half_widths = _initialize_grating(config)

    # 2. Bundle fixed simulation parameters
    fw_initial = 1 / 3
    sim_params = SimulationParameters(
        kappa_array=kappa_array,
        b_initial=jnp.array([jnp.sqrt(fw_initial), jnp.sqrt(fw_initial * 2), 0.0], dtype=jnp.complex64),
    )

    # 3. Create the specialized loss and spectrum functions
    regularization_fn = REGULARIZATION_FNS.get(config.regularization) if config.regularization else None
    loss_fn, spectrum_fn = make_simulation_fns(sim_params, config, regularization_fn, config.lambda_val)

    # 4. Run the optimization on the first half of the parameters
    optimized_half_widths = _run_optimization(initial_half_widths, loss_fn, config)

    # 5. Visualize the results
    print("4. Visualizing optimization results...")
    initial_widths = jnp.concatenate([initial_half_widths, initial_half_widths[::-1]])
    optimized_widths = jnp.concatenate([optimized_half_widths, optimized_half_widths[::-1]])

    _plot_domain_widths(initial_widths, optimized_widths)
    _plot_thw_power_spectrum(initial_widths, optimized_widths, spectrum_fn, config)


if __name__ == "__main__":
    main()
