import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from scipy.integrate import solve_ivp

from qpm import cwes2, mgoslt

jax.config.update("jax_enable_x64", val=True)
memory = Memory(location=".cache", verbose=0)


@dataclass
class SimulationConfig:
    shg_len: float = 5000.0
    sfg_len: float = 5000.0
    kappa_shg_base: float = 1.5e-4 / (2 / jnp.pi)
    temperature: float = 70.0
    wavelength: float = 1.064
    input_power: float = 10.0
    block_size: int = 100


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


@dataclass
class SimulationResult:
    z: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    a3: np.ndarray
    total_power: np.ndarray


def setup_structure(config: SimulationConfig) -> SimulationStructure:
    dk_shg = mgoslt.calc_twm_delta_k(config.wavelength, config.wavelength, config.temperature)
    dk_sfg = mgoslt.calc_twm_delta_k(config.wavelength, config.wavelength / 2, config.temperature)

    widths = jnp.concatenate(
        [
            jnp.full(int(config.shg_len / jnp.abs(jnp.pi / dk_shg)), jnp.abs(jnp.pi / dk_shg)),
            jnp.full(int(config.sfg_len / jnp.abs(jnp.pi / dk_sfg)), jnp.abs(jnp.pi / dk_sfg)),
        ]
    )

    sign_pattern = jnp.array([1.0 if i % 2 == 0 else -1.0 for i in range(len(widths))])

    return SimulationStructure(
        domain_widths=widths,
        kappa_shg_vals=config.kappa_shg_base * sign_pattern,
        kappa_sfg_vals=2 * config.kappa_shg_base * sign_pattern,
        dk_shg=dk_shg,
        dk_sfg=dk_sfg,
        p_in=jnp.array([jnp.sqrt(config.input_power), 0.0, 0.0], dtype=jnp.complex128),
        z_coords=jnp.concatenate([jnp.array([0.0]), jnp.cumsum(widths)]),
        block_size=config.block_size,
    )


def interpolate_polar(z_out: np.ndarray, z_in: np.ndarray, a_in: np.ndarray) -> np.ndarray:
    return np.interp(z_out, z_in, np.abs(a_in)) * np.exp(1j * np.interp(z_out, z_in, np.unwrap(np.angle(a_in))))


def process_trace(struct: SimulationStructure, trace: jax.Array, elapsed: float) -> tuple[SimulationResult, float]:
    trace_np, z_full = np.array(trace), np.array(struct.z_coords)
    z_idx = list(range(0, len(z_full), struct.block_size))
    if z_idx[-1] != len(z_full) - 1:
        z_idx.append(len(z_full) - 1)

    z_sparse = z_full[z_idx[: len(trace_np)]]  # Handle potential length mismatch
    a = [interpolate_polar(z_full, z_sparse, trace_np[:, i]) for i in range(3)]

    # Power conservation
    p_target = np.interp(z_full, z_sparse, np.sum(np.abs(trace_np) ** 2, axis=1))
    correction = np.sqrt(p_target / (np.sum(np.abs(a) ** 2, axis=0) + 1e-20))
    a = [ai * correction for ai in a]

    return SimulationResult(z_full, *a, p_target), elapsed


def run_jax_sim(struct: SimulationStructure, sim_func: Callable) -> tuple[SimulationResult, float]:
    args = (struct.domain_widths, struct.kappa_shg_vals, struct.kappa_sfg_vals, struct.dk_shg, struct.dk_sfg, struct.p_in, struct.block_size)
    print("Warming up JIT...")
    sim_func(*args)[1].block_until_ready()
    print("Warmup complete.")

    st = time.perf_counter()
    _, trace = sim_func(*args)
    trace.block_until_ready()
    return process_trace(struct, trace, time.perf_counter() - st)


@memory.cache
def _solve_ode_core(z, k_shg, k_sfg, dk_shg, dk_sfg, y0, method, rtol, atol):
    def odes(t, A):
        A1, A2, A3 = A
        idx = np.clip(np.searchsorted(z, t, side="right") - 1, 0, len(k_shg) - 1)
        phase_shg, phase_sfg = np.exp(1j * dk_shg * t), np.exp(1j * dk_sfg * t)
        return [
            1j * (k_shg[idx] * A2 * np.conj(A1) * phase_shg + k_sfg[idx] * A3 * np.conj(A2) * phase_sfg),
            1j * (k_shg[idx] * A1**2 * np.conj(phase_shg) + 2 * k_sfg[idx] * A3 * np.conj(A1) * phase_sfg),
            1j * (3 * k_sfg[idx] * A1 * A2 * np.conj(phase_sfg)),
        ]

    kwargs = {}
    if rtol is not None:
        kwargs["rtol"] = rtol
    if atol is not None:
        kwargs["atol"] = atol
    sol = solve_ivp(odes, (z[0], z[-1]), y0, t_eval=z, method=method, **kwargs)
    return sol.t, sol.y, sol.nfev


def run_scipy(struct: SimulationStructure) -> tuple[SimulationResult, float]:
    st = time.perf_counter()
    t, y, nfev = _solve_ode_core(
        np.array(struct.z_coords),
        np.array(struct.kappa_shg_vals),
        np.array(struct.kappa_sfg_vals),
        float(struct.dk_shg),
        float(struct.dk_sfg),
        np.array(struct.p_in, dtype=np.complex128),
        "DOP853",
        1e-10,
        1e-10,
    )
    print(nfev, "(DOP853)")
    return SimulationResult(t, y[0], y[1], y[2], np.sum(np.abs(y) ** 2, axis=0)), time.perf_counter() - st


def plot_comparison(res_map: list[tuple[SimulationResult, str, str]], struct: SimulationStructure, filename: str) -> None:
    plt.rcParams.update({"legend.fontsize": 16})
    # Reverted to 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(20, 20), sharex=True)
    metrics = [("a1", "|A1|"), ("a2", "|A2|"), ("a3", "|A3|"), ("total_power", "Total Power")]

    # Extract kappa base values (magnitude) for the legend/title
    k_shg_mag = jnp.abs(struct.kappa_shg_vals[0])
    k_sfg_mag = jnp.abs(struct.kappa_sfg_vals[0])

    # Create a single legend entry for Kappa using a dummy line
    # We add this to the first axis (A1 plot) or the top of the figure
    kappa_label = f"$\\kappa_{{SHG}}={k_shg_mag:.2e}, \\kappa_{{SFG}}={k_sfg_mag:.2e}$"

    # Add the kappa info to the figure title
    fig.suptitle(f"Simulation Comparison\n({kappa_label})", fontsize=28)

    for ax, (attr, label) in zip(axes, metrics, strict=False):
        for res, name, style in res_map:
            data = getattr(res, attr) if attr == "total_power" else np.abs(getattr(res, attr))
            ax.plot(res.z, data, label=name, linestyle=style, linewidth=3)

        ax.set_ylabel(label, fontsize=24)
        ax.legend(loc="upper right")
        ax.grid(visible=True, linestyle=":")

    axes[-1].set_xlabel("Position", fontsize=24)
    plt.savefig(filename)
    print(f"Plot saved to {filename}")


def main() -> None:
    config = SimulationConfig()

    print("Running Magnus ...")
    print("Running Super Step ...")
    struct_ss = setup_structure(config)
    struct_ss.block_size = 100
    magnus_res, t_magnus = run_jax_sim(struct_ss, cwes2.simulate_magnus_with_trace)
    print(f"Magnus time: {t_magnus:.6f} s")

    print("Running Super Step ...")
    struct_mc = setup_structure(config)
    struct_mc.block_size = 100
    lfaga_res, t_lfaga = run_jax_sim(struct_mc, cwes2.simulate_lfaga_with_trace)
    print(f"Super Step time: {t_lfaga:.6f} s")

    print("Running SciPy DOP853...")
    ver_res, t_ver = run_scipy(struct_mc)
    print(f"SciPy DOP853 time: {t_ver:.6f} s")

    plot_comparison(
        [
            (magnus_res, f"Magnus (BS={struct_ss.block_size}, {t_magnus:.3f}s)", "-"),
            (lfaga_res, f"Super Step (BS={struct_mc.block_size}, {t_lfaga:.3f}s)", "--"),
            (ver_res, "DOP853", ":"),
        ],
        struct_mc,
        "amp_trace_comparison.png",
    )


if __name__ == "__main__":
    main()
