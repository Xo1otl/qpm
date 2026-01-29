import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Adjust imports based on your actual file structure
from qpm import cwes2, mgoslt

# Ensure JAX precision settings
jax.config.update("jax_enable_x64", val=True)


@dataclass
class SimulationConfig:
    shg_len: float = 15000.0
    sfg_len: float = 15000.0
    # Base kappa normalized by 2/pi
    kappa_shg_base: float = 1.5e-5 / (2 / jnp.pi)
    temperature: float = 70.0
    wavelength: float = 1.064
    input_power: float = 10.0


@dataclass
class SimulationStructure:
    domain_widths: jax.Array
    kappa_shg_vals: jax.Array
    kappa_sfg_vals: jax.Array
    dk_shg: jax.Array
    dk_sfg: jax.Array
    p_in: jax.Array
    z_coords: jax.Array


@dataclass
class SimulationResult:
    z: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    a3: np.ndarray
    total_power: np.ndarray


def setup_structure(config: SimulationConfig) -> SimulationStructure:
    """Calculates physical parameters and constructs the domain structure."""
    kappa_sfg = 2 * config.kappa_shg_base

    dk_shg = mgoslt.calc_twm_delta_k(config.wavelength, config.wavelength, config.temperature)
    dk_sfg = mgoslt.calc_twm_delta_k(config.wavelength, config.wavelength / 2, config.temperature)

    # Coherence lengths
    lc_shg = jnp.abs(jnp.pi / dk_shg)
    lc_sfg = jnp.abs(jnp.pi / dk_sfg)

    # Domain widths setup
    n_shg = int(config.shg_len / lc_shg)
    widths_shg = jnp.full(n_shg, lc_shg)

    n_sfg = int(config.sfg_len / lc_sfg)
    widths_sfg = jnp.full(n_sfg, lc_sfg)

    domain_widths = jnp.concatenate([widths_shg, widths_sfg])
    num_domains = len(domain_widths)

    # QPM Sign pattern (+1, -1, +1, ...)
    sign_pattern = jnp.where(jnp.arange(num_domains) % 2 == 0, 1.0, -1.0)

    kappa_shg_vals = config.kappa_shg_base * sign_pattern
    kappa_sfg_vals = kappa_sfg * sign_pattern

    # Initial conditions: [A1, A2, A3]
    p_in = jnp.array([jnp.sqrt(config.input_power), 0.0, 0.0], dtype=jnp.complex64)

    # Z coordinates represent boundaries (N+1 points for N domains)
    z_coords = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(domain_widths)])

    return SimulationStructure(
        domain_widths=domain_widths,
        kappa_shg_vals=kappa_shg_vals,
        kappa_sfg_vals=kappa_sfg_vals,
        dk_shg=dk_shg,
        dk_sfg=dk_sfg,
        p_in=p_in,
        z_coords=z_coords,
    )


def run_simulation(struct: SimulationStructure) -> tuple[SimulationResult, float]:
    """Runs the simulation and returns the trace without interpolation."""

    # 1. Warmup / Compilation
    print("Warming up JIT...")
    # Trigger JIT compilation (discarding first result)
    cwes2.simulate_twm_with_trace(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
    )[0].block_until_ready()
    print("Warmup complete.")

    # 2. Execution
    start_time = time.perf_counter()
    _, trace = cwes2.simulate_twm_with_trace(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
    )
    trace.block_until_ready()
    elapsed_time = time.perf_counter() - start_time

    # 3. Post-processing
    # Simulation trace already returns values at every domain boundary.
    # trace shape: (num_domains + 1, 3)
    trace_np = np.array(trace)
    z = np.array(struct.z_coords)

    a1, a2, a3 = trace_np[:, 0], trace_np[:, 1], trace_np[:, 2]
    total_power = np.abs(a1) ** 2 + np.abs(a2) ** 2 + np.abs(a3) ** 2

    return SimulationResult(z=z, a1=a1, a2=a2, a3=a3, total_power=total_power), elapsed_time


def plot_combined_waves(res: SimulationResult, filename: str = "wave_evolution.png") -> None:
    """Plots A1, A2, and A3 on a single graph. DO NOT MODIFY AS PER USER REQUEST."""
    plt.rcParams.update({"font.size": 14, "legend.fontsize": 12})

    plt.figure(figsize=(10, 6))

    plt.plot(res.z, np.abs(res.a1), color="red", label=r"$A_{\omega}$", linewidth=2)
    plt.plot(res.z, np.abs(res.a2), color="green", label=r"$A_{2\omega}$", linewidth=2)
    plt.plot(res.z, np.abs(res.a3), color="purple", label=r"$A_{3\omega}$", linewidth=2)

    plt.xlabel(r"Position $z$ ($\mu$m)")
    plt.ylabel("Amplitude |A|")
    plt.title("Wave Amplitude Evolution")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Plot saved to {filename}")


def main() -> None:
    config = SimulationConfig()

    # 1. Setup
    struct = setup_structure(config)
    print(f"Structure setup complete. Total domains: {len(struct.domain_widths)}")

    # 2. Run Simulation
    print("Running Simulation Pulse Trace...")
    result, duration = run_simulation(struct)
    print(f"Simulation finished in {duration:.6f} s")

    # 3. Plot
    plot_combined_waves(result)


if __name__ == "__main__":
    main()
