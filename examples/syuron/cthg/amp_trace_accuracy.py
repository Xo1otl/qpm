import time
from dataclasses import dataclass
from typing import Any

import jax

jax.config.update("jax_enable_x64", val=False)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from qpm import cwes2, mgoslt


@dataclass
class SimulationConfig:
    shg_len: float = 10000.0
    sfg_len: float = 30000.0
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
    power_deviation_ratio: np.ndarray


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


def calculate_power_stats(a1: np.ndarray, a2: np.ndarray, a3: np.ndarray, initial_power: float) -> tuple[np.ndarray, np.ndarray]:
    # Taking abs^2 for power
    p1 = np.abs(a1) ** 2
    p2 = np.abs(a2) ** 2
    p3 = np.abs(a3) ** 2

    total_power = p1 + p2 + p3
    deviation = (total_power - initial_power) / initial_power
    return total_power, deviation


def run_perturbation(struct: SimulationStructure) -> tuple[SimulationResult, float]:
    # Warmup
    print("Warming up JIT...")
    cwes2.simulate_twm_with_trace(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
    )[1].block_until_ready()
    print("Warmup complete.")
    start_time = time.perf_counter()
    _, trace = cwes2.simulate_twm_with_trace(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
    )
    # Ensure computation is done before stopping timer
    trace.block_until_ready()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # trace is jax array, convert to numpy
    # Assuming trace shape is (N, 3) for the 3 fields
    trace_np = np.array(trace)
    a1 = trace_np[:, 0]
    a2 = trace_np[:, 1]
    a3 = trace_np[:, 2]

    z = np.array(struct.z_coords)
    initial_power = np.abs(struct.p_in[0]) ** 2 + np.abs(struct.p_in[1]) ** 2 + np.abs(struct.p_in[2]) ** 2

    total_power, deviation = calculate_power_stats(a1, a2, a3, float(initial_power))

    return SimulationResult(z=z, a1=a1, a2=a2, a3=a3, total_power=total_power, power_deviation_ratio=deviation), elapsed_time


def run_scipy_ode(struct: SimulationStructure, *, verification: bool = True) -> tuple[SimulationResult, float]:
    z_coords = np.array(struct.z_coords)
    kappa_shg_vals = np.array(struct.kappa_shg_vals)
    kappa_sfg_vals = np.array(struct.kappa_sfg_vals)
    dk_shg = float(struct.dk_shg)
    dk_sfg = float(struct.dk_sfg)

    y0 = np.array(struct.p_in, dtype=np.complex128)

    def odes(z: float, A: tuple[float, float, float]) -> list[Any]:
        A1, A2, A3 = A
        idx = np.searchsorted(z_coords, z, side="right") - 1
        idx = np.clip(idx, 0, len(kappa_shg_vals) - 1)
        k_shg = kappa_shg_vals[idx]
        k_sfg = kappa_sfg_vals[idx]

        dA1 = 1j * (k_shg * A2 * np.conj(A1) * np.exp(1j * dk_shg * z) + k_sfg * A3 * np.conj(A2) * np.exp(1j * dk_sfg * z))
        dA2 = 1j * (k_shg * A1**2 * np.exp(-1j * dk_shg * z) + 2 * k_sfg * A3 * np.conj(A1) * np.exp(1j * dk_sfg * z))
        dA3 = 1j * (3 * k_sfg * A1 * A2 * np.exp(-1j * dk_sfg * z))
        return [dA1, dA2, dA3]

    start_time = time.perf_counter()
    if verification:
        sol = solve_ivp(
            odes,
            t_span=(z_coords[0], z_coords[-1]),
            y0=y0,
            t_eval=z_coords,
            method="DOP853",
            rtol=1e-6,
            atol=1e-6,
        )
    else:
        sol = solve_ivp(
            odes,
            t_span=(z_coords[0], z_coords[-1]),
            y0=y0,
            t_eval=z_coords,
            method="DOP853",
            rtol=1e-5,
            atol=1e-5,
        )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    a1 = sol.y[0]
    a2 = sol.y[1]
    a3 = sol.y[2]
    print(sol.nfev)

    initial_power = np.abs(y0[0]) ** 2 + np.abs(y0[1]) ** 2 + np.abs(y0[2]) ** 2
    total_power, deviation = calculate_power_stats(a1, a2, a3, float(initial_power))

    return SimulationResult(z=z_coords, a1=a1, a2=a2, a3=a3, total_power=total_power, power_deviation_ratio=deviation), elapsed_time


def plot_results(
    pert_res: SimulationResult,
    scipy_res: SimulationResult,
    scipy_res_verification: SimulationResult,
    time_pert: float,
    time_scipy: float,
    time_scipy_ver: float,
    filename: str = "amp_trace_comparison.png",
) -> None:
    # Plot Amplitudes (A3)
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(pert_res.z, np.abs(pert_res.a3), label=rf"$3\omega$ (Perturbation, {time_pert:.4f}s)", linewidth=2)
    plt.plot(scipy_res.z, np.abs(scipy_res.a3), label=rf"$3\omega$ (SciPy RK45, {time_scipy:.4f}s)", linewidth=2)
    plt.plot(scipy_res_verification.z, np.abs(scipy_res_verification.a3), label=rf"$3\omega$ (SciPy DOP853, {time_scipy_ver:.4f}s)", linewidth=2)
    plt.xlabel(r"Position ($\mu m$)")
    plt.ylabel("Amplitude")
    plt.title("Comparison of $a_3$ Amplitude")
    plt.legend()
    plt.grid(visible=True, linestyle=":")

    # Plot Power Deviation (Manley-Rowe)
    plt.subplot(2, 1, 2)
    plt.plot(pert_res.z, pert_res.power_deviation_ratio, label="Perturbation", linewidth=2)
    plt.plot(scipy_res.z, scipy_res.power_deviation_ratio, label="SciPy RK45", linewidth=2)
    plt.plot(scipy_res_verification.z, scipy_res_verification.power_deviation_ratio, label="SciPy DOP853", linewidth=2)
    plt.xlabel(r"Position ($\mu m$)")
    plt.ylabel("Relative Power Deviation $(P - P_0)/P_0$")
    plt.title("Manley-Rowe Relation Deviation")
    plt.legend()
    plt.grid(visible=True, linestyle=":")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")


def main() -> None:
    config = SimulationConfig()
    # 1. Setup
    struct = setup_structure(config)
    print(f"Structure setup complete. Num domains: {len(struct.domain_widths)}")

    # 2. Perturbation
    print("Running Perturbation...")
    pert_res, time_pert = run_perturbation(struct)
    print(f"Perturbation time: {time_pert:.6f} s")

    # 3. SciPy (RK45)
    print("Running SciPy DOP853 (tol=1e-5)...")
    scipy_res, time_scipy = run_scipy_ode(struct, verification=False)
    print(f"SciPy RK45 time:   {time_scipy:.6f} s")

    # 4. SciPy (DOP853)
    print("Running SciPy DOP853 (tol=1e-7)...")
    scipy_res_verification, time_scipy_ver = run_scipy_ode(struct, verification=True)
    print(f"SciPy DOP853 time: {time_scipy_ver:.6f} s")

    # 5. Plot
    plot_results(pert_res, scipy_res, scipy_res_verification, time_pert, time_scipy, time_scipy_ver)


if __name__ == "__main__":
    main()
