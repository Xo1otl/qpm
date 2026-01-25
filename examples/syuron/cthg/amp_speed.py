import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
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
    block_size: int = 25


@dataclass
class SimulationStructure:
    domain_widths: jax.Array
    kappa_shg_vals: jax.Array
    kappa_sfg_vals: jax.Array
    dk_shg: jax.Array
    dk_sfg: jax.Array
    p_in: jax.Array
    z_coords: jax.Array
    block_size: int


def setup_structure(config: SimulationConfig) -> SimulationStructure:
    kappa_sfg = 2 * config.kappa_shg_base

    dk_shg = mgoslt.calc_twm_delta_k(config.wavelength, config.wavelength, config.temperature)
    dk_sfg = mgoslt.calc_twm_delta_k(config.wavelength, config.wavelength / 2, config.temperature)

    lc_shg = jnp.pi / dk_shg
    lc_sfg = jnp.pi / dk_sfg

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
        block_size=config.block_size,
    )


def run_perturbation(struct: SimulationStructure) -> tuple[jax.Array, float]:
    print("Warming up JIT...")
    cwes2.simulate_super_step(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
        struct.block_size,
    ).block_until_ready()
    print("Warmup complete.")
    start_time = time.time()
    b_final = cwes2.simulate_super_step(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
        struct.block_size,
    )
    b_final.block_until_ready()
    end_time = time.time()
    return b_final[2], end_time - start_time


def run_npda(struct: SimulationStructure) -> tuple[jax.Array, float]:
    print("Warming up JIT...")
    cwes2.calc_a3_npda(
        struct.p_in[0],
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.domain_widths,
        struct.dk_shg,
        struct.dk_sfg,
    ).block_until_ready()
    print("Warmup complete.")
    start_time = time.time()
    a1 = struct.p_in[0]
    a3_final = cwes2.calc_a3_npda(
        a1,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.domain_widths,
        struct.dk_shg,
        struct.dk_sfg,
    )
    a3_final.block_until_ready()
    end_time = time.time()
    return a3_final, end_time - start_time


def run_scipy_ode(
    struct: SimulationStructure,
    method: str = "RK45",
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> tuple[np.ndarray, float]:
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
        t_eval=[z_coords[-1]],  # Only evaluate at the end
        method=method,
        rtol=rtol,
        atol=atol,
    )

    a3_scipy = np.abs(sol.y[2][-1])
    end_time = time.time()
    return a3_scipy, end_time - start_time


def main() -> None:
    config = SimulationConfig()
    struct = setup_structure(config)

    print(f"Structure setup complete. Num domains: {len(struct.domain_widths)}")

    # 1. Perturbation Simulation
    a3_pert_val, time_pert = run_perturbation(struct)
    a3_pert = jnp.abs(a3_pert_val)
    print(f"Perturbation simulation time: {time_pert:.6f} s, |a3| = {a3_pert:.6e}")

    # 2. NPDA Simulation
    a3_npda_val, time_npda = run_npda(struct)
    a3_npda = jnp.abs(a3_npda_val)
    print(f"NPDA calculation time:        {time_npda:.6f} s, |a3| = {a3_npda:.6e}")

    # 3. SciPy ODE Simulation (RK45)
    a3_scipy, time_scipy = run_scipy_ode(struct)
    print(f"SciPy ODE (RK45) time:        {time_scipy:.6f} s, |a3| = {a3_scipy:.6e}")

    # 4. SciPy ODE Simulation (Ground Truth)
    a3_gt, time_gt = run_scipy_ode(struct, method="DOP853", rtol=1e-8, atol=1e-8)
    print(f"SciPy ODE (DOP853/GT) time:   {time_gt:.6f} s, |a3| = {a3_gt:.6e}")


if __name__ == "__main__":
    main()
