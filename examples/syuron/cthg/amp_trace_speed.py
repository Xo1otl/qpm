import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from qpm import cwes2, mgoslt


@dataclass
class SimulationConfig:
    shg_len: float = 10000.0
    sfg_len: float = 7500.0
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


def setup_structure(config: SimulationConfig) -> SimulationStructure:
    kappa_sfg = 2 * config.kappa_shg_base

    dk_shg = mgoslt.calc_twm_delta_k(config.wavelength, config.wavelength, config.temperature)
    dk_sfg = mgoslt.calc_twm_delta_k(config.wavelength, config.wavelength / 2, config.temperature)

    lc_shg = jnp.abs(jnp.pi / dk_shg)
    lc_sfg = jnp.abs(jnp.pi / dk_sfg)

    n_shg = int(config.shg_len / lc_shg)
    widths_shg = jnp.full(n_shg, lc_shg)

    n_sfg = int(config.sfg_len / lc_sfg)
    widths_sfg = jnp.full(n_sfg, lc_sfg)

    domain_widths = jnp.concatenate([widths_shg, widths_sfg])
    num_domains = len(domain_widths)

    sign_pattern = jnp.array([1.0 if i % 2 == 0 else -1.0 for i in range(num_domains)])
    kappa_shg_vals = config.kappa_shg_base * sign_pattern
    kappa_sfg_vals = kappa_sfg * sign_pattern

    p_in = jnp.array([jnp.sqrt(config.input_power), 0.0, 0.0], dtype=jnp.complex64)
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


def run_perturbation(struct: SimulationStructure) -> tuple[jax.Array, float]:
    start_time = time.time()
    _, trace = cwes2.simulate_twm_with_trace(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
    )
    trace.block_until_ready()
    end_time = time.time()
    return trace, end_time - start_time


def run_npda(struct: SimulationStructure) -> tuple[jax.Array, float]:
    start_time = time.time()
    a1 = struct.p_in[0]
    a3_trace = cwes2.calc_a3_npda_trace(
        a1,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.domain_widths,
        struct.dk_shg,
        struct.dk_sfg,
    )
    a3_trace.block_until_ready()
    end_time = time.time()
    return a3_trace, end_time - start_time


def run_scipy_ode(struct: SimulationStructure) -> tuple[np.ndarray, float]:
    # Convert JAX arrays to NumPy for SciPy solver
    z_coords = np.array(struct.z_coords)
    kappa_shg_vals = np.array(struct.kappa_shg_vals)
    kappa_sfg_vals = np.array(struct.kappa_sfg_vals)
    dk_shg = float(struct.dk_shg)
    dk_sfg = float(struct.dk_sfg)

    # Initial conditions
    y0 = np.array(struct.p_in, dtype=np.complex128)

    def odes(z: float, A: tuple[float, float, float]) -> list[Any]:
        A1, A2, A3 = A

        # Identify the current domain index
        idx = np.searchsorted(z_coords, z, side="right") - 1
        idx = np.clip(idx, 0, len(kappa_shg_vals) - 1)

        k_shg = kappa_shg_vals[idx]
        k_sfg = kappa_sfg_vals[idx]

        dA1 = 1j * (k_shg * A2 * np.conj(A1) * np.exp(1j * dk_shg * z) + k_sfg * A3 * np.conj(A2) * np.exp(1j * dk_sfg * z))
        dA2 = 1j * (k_shg * A1**2 * np.exp(-1j * dk_shg * z) + 2 * k_sfg * A3 * np.conj(A1) * np.exp(1j * dk_sfg * z))
        dA3 = 1j * (3 * k_sfg * A1 * A2 * np.exp(-1j * dk_sfg * z))

        return [dA1, dA2, dA3]

    # Solve the ODE
    start_time = time.time()
    sol = solve_ivp(
        odes,
        t_span=(z_coords[0], z_coords[-1]),
        y0=y0,
        t_eval=z_coords,
        method="RK45",
    )

    a3_scipy = np.abs(sol.y[2])
    end_time = time.time()
    return a3_scipy, end_time - start_time


def main() -> None:
    config = SimulationConfig()
    struct = setup_structure(config)

    print(f"Structure setup complete. Num domains: {len(struct.domain_widths)}")

    # Warmup (trigger JIT compilation)
    print("Warming up JIT...")
    cwes2.simulate_twm_with_trace(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
    )[1].block_until_ready()
    cwes2.calc_a3_npda_trace(
        struct.p_in[0],
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.domain_widths,
        struct.dk_shg,
        struct.dk_sfg,
    ).block_until_ready()
    print("Warmup complete.")

    # 1. Perturbation Simulation
    trace_pert, time_pert = run_perturbation(struct)
    a3_pert = jnp.abs(trace_pert[:, 2])
    print(f"Perturbation simulation time: {time_pert:.6f} s")

    # 2. NPDA Simulation
    trace_npda, time_npda = run_npda(struct)
    a3_npda = jnp.abs(trace_npda)
    print(f"NPDA calculation time:        {time_npda:.6f} s")

    # 3. SciPy ODE Simulation
    a3_scipy, time_scipy = run_scipy_ode(struct)
    print(f"SciPy ODE calculation time:   {time_scipy:.6f} s")

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot Perturbation Result
    plt.plot(struct.z_coords, a3_pert, label=r"$3\omega$ (Perturbation)", linewidth=2)

    # Plot NPDA Result
    plt.plot(struct.z_coords, a3_npda, label=r"$3\omega$ (UPA)", linestyle="--", linewidth=2)

    # Plot SciPy Result
    plt.plot(struct.z_coords, a3_scipy, label=r"$3\omega$ (SciPy)", linestyle=":", linewidth=2)

    plt.xlabel(r"Position ($\mu m$)")
    plt.ylabel("Amplitude")
    plt.title(f"Comparison of $a_3$ Calculation\nPerturbation: {time_pert:.4f}s, NPDA: {time_npda:.4f}s, SciPy: {time_scipy:.4f}s")
    plt.legend()
    plt.grid(visible=True, linestyle=":")

    output_filename = "amp_trace_comparison.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")


if __name__ == "__main__":
    main()
