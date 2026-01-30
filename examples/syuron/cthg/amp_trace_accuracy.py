import time
from dataclasses import dataclass
from typing import Any

import jax

jax.config.update("jax_enable_x64", val=True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from scipy.integrate import solve_ivp

from qpm import cwes2, mgoslt

memory = Memory(location=".cache", verbose=0)


@dataclass
class SimulationConfig:
    shg_len: float = 10000.0
    sfg_len: float = 7500.0
    kappa_shg_base: float = 1.5e-5 / (2 / jnp.pi)
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

    p_in = jnp.array([jnp.sqrt(config.input_power), 0.0, 0.0], dtype=jnp.complex128)
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


def calculate_mse(res: SimulationResult, ref: SimulationResult) -> tuple[float, float, float]:
    mse1 = np.mean((np.abs(res.a1) - np.abs(ref.a1)) ** 2)
    mse2 = np.mean((np.abs(res.a2) - np.abs(ref.a2)) ** 2)
    mse3 = np.mean((np.abs(res.a3) - np.abs(ref.a3)) ** 2)
    return float(mse1), float(mse2), float(mse3)


def run_perturbation(struct: SimulationStructure) -> tuple[SimulationResult, float]:
    # Warmup
    print("Warming up JIT...")
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
    # Ensure computation is done before stopping timer
    trace.block_until_ready()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # trace is jax array, convert to numpy
    # Assuming trace shape is (N_blocks + 1, 3)
    trace_np = np.array(trace)
    a1_sparse = trace_np[:, 0]
    a2_sparse = trace_np[:, 1]
    a3_sparse = trace_np[:, 2]

    z_full = np.array(struct.z_coords)
    # Construct z_sparse corresponding to the trace points.
    # Trace points are at 0, block_size, 2*block_size, ...
    # If the last block is padded, the final trace point corresponds to the end of the structure (z_full[-1]).
    z_indices = list(range(0, len(z_full), struct.block_size))
    if z_indices[-1] != len(z_full) - 1:
        z_indices.append(len(z_full) - 1)

    # Verify exact length match just in case
    if len(z_indices) != len(a1_sparse):
        # Fallback or error if assumption fails (though logic holds for standard padding)
        # Assuming just taking bounds if length differs slightly (e.g. if one extra point due to scan behavior?)
        # trace should have N_blocks + 1. z_indices has N_blocks + 1.
        print(f"Warning: Trace length {len(a1_sparse)} vs Indices length {len(z_indices)}. adjusting...")
        z_sparse = z_full[z_indices[: len(a1_sparse)]]
    else:
        z_sparse = z_full[z_indices]

    # Interpolate to full grid using Polar Interpolation (Magnitude & Phase)
    # This prevents magnitude dips ("scalloping") between sparse points when phase rotates.

    def interpolate_polar(z_out: np.ndarray, z_in: np.ndarray, a_in: np.ndarray) -> np.ndarray:
        mag = np.abs(a_in)
        phase = np.unwrap(np.angle(a_in))

        mag_out = np.interp(z_out, z_in, mag)
        phase_out = np.interp(z_out, z_in, phase)
        return mag_out * np.exp(1j * phase_out)

    a1 = interpolate_polar(z_full, z_sparse, a1_sparse)
    a2 = interpolate_polar(z_full, z_sparse, a2_sparse)
    a3 = interpolate_polar(z_full, z_sparse, a3_sparse)

    total_power = np.abs(a1) ** 2 + np.abs(a2) ** 2 + np.abs(a3) ** 2

    return SimulationResult(z=z_full, a1=a1, a2=a2, a3=a3, total_power=total_power), elapsed_time


@memory.cache
def _solve_ode_core(  # noqa: PLR0913
    z_coords: np.ndarray,
    kappa_shg_vals: np.ndarray,
    kappa_sfg_vals: np.ndarray,
    dk_shg: float,
    dk_sfg: float,
    y0: np.ndarray,
    method: str,
    rtol: float | None = None,
    atol: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
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

    kwargs = {}
    if rtol is not None:
        kwargs["rtol"] = rtol
    if atol is not None:
        kwargs["atol"] = atol

    sol = solve_ivp(
        odes,
        t_span=(z_coords[0], z_coords[-1]),
        y0=y0,
        t_eval=z_coords,
        method=method,
        **kwargs,
    )
    return sol.t, sol.y, sol.nfev


def run_scipy_ode(struct: SimulationStructure, *, verification: bool = True) -> tuple[SimulationResult, float]:
    z_coords = np.array(struct.z_coords)
    kappa_shg_vals = np.array(struct.kappa_shg_vals)
    kappa_sfg_vals = np.array(struct.kappa_sfg_vals)
    dk_shg = float(struct.dk_shg)
    dk_sfg = float(struct.dk_sfg)

    y0 = np.array(struct.p_in, dtype=np.complex128)

    method = "DOP853" if verification else "RK45"
    rtol = 1e-8 if verification else None
    atol = 1e-8 if verification else None

    start_time = time.perf_counter()
    t, y, nfev = _solve_ode_core(z_coords, kappa_shg_vals, kappa_sfg_vals, dk_shg, dk_sfg, y0, method, rtol, atol)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    a1 = y[0]
    a2 = y[1]
    a3 = y[2]
    print(nfev)

    total_power = np.abs(a1) ** 2 + np.abs(a2) ** 2 + np.abs(a3) ** 2

    return SimulationResult(z=t, a1=a1, a2=a2, a3=a3, total_power=total_power), elapsed_time


def plot_results_notext(
    pert_res: SimulationResult,
    scipy_res: SimulationResult,
    scipy_res_ver: SimulationResult,
    time_pert: float,
    time_scipy: float,
    time_scipy_ver: float,
    filename: str = "amp_trace_comparison.png",
) -> None:
    # 凡例のフォントサイズを32に設定
    plt.rcParams.update({"legend.fontsize": 32})

    plt.figure(figsize=(20, 15))  # 凡例が大きくなるため、少し横幅を広げました

    # FW (A1)
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(pert_res.z, np.abs(pert_res.a1), label="Super Step", linewidth=3)
    plt.plot(scipy_res.z, np.abs(scipy_res.a1), label="RK45", linewidth=3)
    plt.plot(scipy_res_ver.z, np.abs(scipy_res_ver.a1), label="DOP853", linestyle="--", linewidth=3)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.legend(loc="upper right")
    plt.grid(visible=True, linestyle=":")

    # SHW (A2)
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(pert_res.z, np.abs(pert_res.a2), label="Super Step", linewidth=3)
    plt.plot(scipy_res.z, np.abs(scipy_res.a2), label="RK45", linewidth=3)
    plt.plot(scipy_res_ver.z, np.abs(scipy_res_ver.a2), label="DOP853", linestyle="--", linewidth=3)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    plt.legend(loc="upper right")
    plt.grid(visible=True, linestyle=":")

    # THW (A3)
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(pert_res.z, np.abs(pert_res.a3), label="Super Step", linewidth=3)
    plt.plot(scipy_res.z, np.abs(scipy_res.a3), label="RK45", linewidth=3)
    plt.plot(scipy_res_ver.z, np.abs(scipy_res_ver.a3), label="DOP853", linestyle="--", linewidth=3)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    plt.legend(loc="upper right")
    plt.grid(visible=True, linestyle=":")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename} with large legends (font size 32).")


def plot_results(  # noqa: PLR0913
    pert_res: SimulationResult,
    scipy_res: SimulationResult,
    scipy_res_ver: SimulationResult,
    time_pert: float,
    time_scipy: float,
    time_scipy_ver: float,
    filename: str = "amp_trace_comparison.png",
) -> None:
    # Centralized font size configuration
    scale = 1
    plt.rcParams.update(
        {
            "font.size": 16 * scale,
            "axes.titlesize": 18 * scale,
            "axes.labelsize": 16 * scale,
            "xtick.labelsize": 14 * scale,
            "ytick.labelsize": 14 * scale,
            "legend.fontsize": 14 * scale,
            "figure.titlesize": 20 * scale,
        },
    )

    # Plot Amplitudes (A3)
    # Calculate MSE
    mse_pert = calculate_mse(pert_res, scipy_res_ver)
    mse_scipy = calculate_mse(scipy_res, scipy_res_ver)

    print(f"MSE (Super Step vs DOP853): FW={mse_pert[0]:.2e}, SHW={mse_pert[1]:.2e}, THW={mse_pert[2]:.2e}")
    print(f"MSE (RK45 vs DOP853):         FW={mse_scipy[0]:.2e}, SHW={mse_scipy[1]:.2e}, THW={mse_scipy[2]:.2e}")

    plt.figure(figsize=(15, 12))

    # Plot FW (A1)
    plt.subplot(3, 1, 1)
    plt.plot(pert_res.z, np.abs(pert_res.a1), label=rf"$1\omega$ (Super Step, MSE={mse_pert[0]:.1e})", linewidth=2)
    plt.plot(scipy_res.z, np.abs(scipy_res.a1), label=rf"$1\omega$ (RK45, MSE={mse_scipy[0]:.1e})", linewidth=2)
    plt.plot(scipy_res_ver.z, np.abs(scipy_res_ver.a1), label=r"$1\omega$ (DOP853)", linestyle="--", linewidth=2)
    plt.ylabel("Amplitude")
    plt.title(rf"Fundamental Wave ($1\omega$) : Super Step time {time_pert:.4f}s vs RK45 time {time_scipy:.4f} vs DOP853 {time_scipy_ver:.4f}s")
    plt.legend()
    plt.grid(visible=True, linestyle=":")

    # Plot SHW (A2)
    plt.subplot(3, 1, 2)
    plt.plot(pert_res.z, np.abs(pert_res.a2), label=rf"$2\omega$ (Super Step, MSE={mse_pert[1]:.1e})", linewidth=2)
    plt.plot(scipy_res.z, np.abs(scipy_res.a2), label=rf"$2\omega$ (RK45, MSE={mse_scipy[1]:.1e})", linewidth=2)
    plt.plot(scipy_res_ver.z, np.abs(scipy_res_ver.a2), label=r"$2\omega$ (DOP853)", linestyle="--", linewidth=2)
    plt.ylabel("Amplitude")
    plt.title(r"Second Harmonic Wave ($2\omega$)")
    plt.legend()
    plt.grid(visible=True, linestyle=":")

    # Plot THW (A3)
    plt.subplot(3, 1, 3)
    plt.plot(pert_res.z, np.abs(pert_res.a3), label=rf"$3\omega$ (Super Step, MSE={mse_pert[2]:.1e})", linewidth=2)
    plt.plot(scipy_res.z, np.abs(scipy_res.a3), label=rf"$3\omega$ (RK45, MSE={mse_scipy[2]:.1e})", linewidth=2)
    plt.plot(scipy_res_ver.z, np.abs(scipy_res_ver.a3), label=r"$3\omega$ (DOP853)", linestyle="--", linewidth=2)
    plt.xlabel(r"Position ($\mu m$)")
    plt.ylabel("Amplitude")
    plt.title(r"Third Harmonic Wave ($3\omega$)")
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

    # 2. Super Step
    print("Running Super Step...")
    pert_res, time_pert = run_perturbation(struct)
    print(f"Super Step time: {time_pert:.6f} s")

    # 3. SciPy (RK45)
    print("Running SciPy RK ...")
    scipy_res, time_scipy = run_scipy_ode(struct, verification=False)
    print(f"SciPy RK45 time:   {time_scipy:.6f} s")

    # 4. SciPy (DOP853)
    print("Running SciPy DOP853 ...")
    scipy_res_ver, time_scipy_ver = run_scipy_ode(struct, verification=True)
    print(f"SciPy DOP853 time: {time_scipy_ver:.6f} s")

    # 5. Plot
    plot_results_notext(pert_res, scipy_res, scipy_res_ver, time_pert, time_scipy, time_scipy)


if __name__ == "__main__":
    main()
