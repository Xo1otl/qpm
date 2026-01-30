import time
from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", val=True)

import jax.numpy as jnp
import numpy as np

from qpm import cwes2, mgoslt


@dataclass
class SimulationConfig:
    shg_len: float = 50000.0
    sfg_len: float = 50000.0
    kappa_shg_base: float = 1.5e-5 / (2 / jnp.pi)
    temperature: float = 70.0
    wavelength: float = 1.064
    input_power: float = 10.0
    block_size: int = 1  # Using block size 1 for direct comparison


@dataclass
class SimulationStructure:
    domain_widths: jax.Array
    kappa_shg_vals: jax.Array
    kappa_sfg_vals: jax.Array
    dk_shg: jax.Array
    dk_sfg: jax.Array
    p_in: jax.Array
    block_size: int


@dataclass
class SimulationResult:
    a1: np.ndarray
    a2: np.ndarray
    a3: np.ndarray
    total_power: np.ndarray


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

    # Simple periodic sign pattern for now
    sign_pattern = jnp.array([1.0 if i % 2 == 0 else -1.0 for i in range(num_domains)])
    kappa_shg_vals = config.kappa_shg_base * sign_pattern
    kappa_sfg_vals = kappa_sfg * sign_pattern

    p_in = jnp.array([jnp.sqrt(config.input_power), 0.0, 0.0], dtype=jnp.complex128)

    return SimulationStructure(
        domain_widths=domain_widths,
        kappa_shg_vals=kappa_shg_vals,
        kappa_sfg_vals=kappa_sfg_vals,
        dk_shg=dk_shg,
        dk_sfg=dk_sfg,
        p_in=p_in,
        block_size=config.block_size,
    )


def run_super_step(struct: SimulationStructure) -> tuple[SimulationResult, float]:
    print("Warming up Super Step...")
    cwes2.simulate_super_step_with_trace(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
        struct.block_size,
    )[1].block_until_ready()
    print("Warmup complete.")

    start_time = time.perf_counter()
    _, trace = cwes2.simulate_super_step_with_trace(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
        struct.block_size,
    )
    trace.block_until_ready()
    end_time = time.perf_counter()

    trace_np = np.array(trace)

    # Demodulate B -> A
    # B(z) = exp(i L z) A(z)
    # A(z) = exp(-i L z) B(z)
    # z coordinates for trace
    # trace len = N_blocks + 1
    # z[0] = 0
    # z[k] = sum(widths[:k*block_size])

    # Reconstruct z array matching trace
    z_full = np.concatenate([np.array([0.0]), np.cumsum(struct.domain_widths)])
    # Subsample if block_size > 1
    z_trace = z_full[:: struct.block_size]
    # Ensure length matches
    if len(z_trace) != len(trace_np):
        # If padding happened, trace might include extra block or different logic
        # Here we assume block_size=1 so it fits
        z_trace = z_trace[: len(trace_np)]

    dk_shg = np.array(struct.dk_shg)
    dk_sfg = np.array(struct.dk_sfg)

    phase_shg = np.exp(-1j * dk_shg * z_trace)
    phase_sfg = np.exp(-1j * (dk_shg + dk_sfg) * z_trace)

    a1 = trace_np[:, 0]  # L=0
    a2 = trace_np[:, 1] * phase_shg
    a3 = trace_np[:, 2] * phase_sfg

    return SimulationResult(a1=a1, a2=a2, a3=a3, total_power=np.abs(a1) ** 2 + np.abs(a2) ** 2 + np.abs(a3) ** 2), end_time - start_time


def run_lfaga(struct: SimulationStructure) -> tuple[SimulationResult, float]:
    print("Warming up LFAGA...")
    cwes2.simulate_lfaga_with_trace(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
        struct.block_size,
    )[1].block_until_ready()
    print("Warmup complete.")

    start_time = time.perf_counter()
    _, trace = cwes2.simulate_lfaga_with_trace(
        struct.domain_widths,
        struct.kappa_shg_vals,
        struct.kappa_sfg_vals,
        struct.dk_shg,
        struct.dk_sfg,
        struct.p_in,
        struct.block_size,
    )
    trace.block_until_ready()
    end_time = time.perf_counter()

    trace_np = np.array(trace)
    return SimulationResult(
        a1=trace_np[:, 0],
        a2=trace_np[:, 1],
        a3=trace_np[:, 2],
        total_power=np.abs(trace_np[:, 0]) ** 2 + np.abs(trace_np[:, 1]) ** 2 + np.abs(trace_np[:, 2]) ** 2,
    ), end_time - start_time


def main() -> None:
    config = SimulationConfig()
    struct = setup_structure(config)
    print(f"Structure setup complete. Num domains: {len(struct.domain_widths)}")

    res_ss, time_ss = run_super_step(struct)
    res_lfaga, time_lfaga = run_lfaga(struct)

    # Calculate MSE
    mse1 = np.mean(np.abs(res_ss.a1 - res_lfaga.a1) ** 2)
    mse2 = np.mean(np.abs(res_ss.a2 - res_lfaga.a2) ** 2)
    mse3 = np.mean(np.abs(res_ss.a3 - res_lfaga.a3) ** 2)

    print("-" * 40)
    print("Comparison Results:")
    print(f"Super Step Time: {time_ss:.6f} s")
    print(f"LFAGA Time:      {time_lfaga:.6f} s")
    print("-" * 40)
    print(f"MSE A1 (Fundamental): {mse1:.6e}")
    print(f"MSE A2 (SHG):         {mse2:.6e}")
    print(f"MSE A3 (THG):         {mse3:.6e}")
    print("-" * 40)
    print(f"Final Power SS:    {res_ss.total_power[-1]:.6f}")
    print(f"Final Power LFAGA: {res_lfaga.total_power[-1]:.6f}")

    # Check for large discrepancy
    if mse3 > 1e-4:  # Arbitrary threshold, tune as needed
        print("WARNING: Significant discrepancy detected!")
    else:
        print("Success: Results match closely.")


if __name__ == "__main__":
    main()
