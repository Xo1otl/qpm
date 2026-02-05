import time
from dataclasses import dataclass
from typing import Any, Tuple

# jax config is usually global, but safe to set here or in main. 
# Original file set it at top level. I will set it in main.py to avoid side effects on import, 
# or keep it if needed for JIT compilation at import time? 
# Only needed for cwes2 which is already imported.
import numpy as np
from joblib import Memory
from scipy.integrate import solve_ivp

from qpm import cwes2
from structure import SimulationStructure

memory = Memory(location=".cache", verbose=0)


@dataclass
class SimulationResult:
    z: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    a3: np.ndarray
    total_power: np.ndarray


def calculate_mse(res: SimulationResult, ref: SimulationResult) -> Tuple[float, float, float]:
    """Calculate MSE between result and reference. 
    Note: 'res' and 'ref' might have different z-grids. 
    If this is for verification, we usually want to compare at same z.
    However, Magnus result is now sparse. 
    We should interpolate Magnus result to Ref grid for MSE calculation, 
    OR interpolate Ref to Magnus grid?
    Usually Ref (DOP853) is the ground truth.
    Original code calculated MSE on 'interpolated' Magnus result.
    Since we removed interpolation in run_magnus, we must interpolate HERE if we want meaningful MSE.
    """
    # Interpolate res (sparse/whatever) to match ref.z
    # But wait, if res.z is sparse, we can interpolate ref to res.z?
    # Or interpolate res to ref.z?
    # If res is sparse (Magnus), we probably want to check accuracy at those steps.
    # So let's interpolate Ref to Res.z locations.
    
    # But original code computed MSE on the dense grid? 
    # "mse1 = np.mean((np.abs(res.a1) - np.abs(ref.a1)) ** 2)"
    # If shapes mismatch, this fails.
    # Since we changed Magnus to be sparse, we must handle this.
    # I will interpolate Ref to Res.z for MSE calculation if shapes differ.
    
    a1_ref = ref.a1
    a2_ref = ref.a2
    a3_ref = ref.a3
    
    if res.z.shape != ref.z.shape or not np.allclose(res.z, ref.z):
        # Interpolate ref to res.z
        # Note: ref.z is likely fine grid.
        # We want to know error of res at its points.
        a1_ref = np.interp(res.z, ref.z, np.abs(ref.a1)) # Magnitude check? Original used abs.
        a2_ref = np.interp(res.z, ref.z, np.abs(ref.a2))
        a3_ref = np.interp(res.z, ref.z, np.abs(ref.a3))
        # Note: Original MSE was on magnitudes: np.mean((np.abs(res.a1) - np.abs(ref.a1)) ** 2)
        # So interpolating magnitudes is correct for that metric.
    else:
        a1_ref = np.abs(a1_ref)
        a2_ref = np.abs(a2_ref)
        a3_ref = np.abs(a3_ref)

    mse1 = np.mean((np.abs(res.a1) - a1_ref) ** 2)
    mse2 = np.mean((np.abs(res.a2) - a2_ref) ** 2)
    mse3 = np.mean((np.abs(res.a3) - a3_ref) ** 2)
    return float(mse1), float(mse2), float(mse3)


def run_magnus(struct: SimulationStructure) -> Tuple[SimulationResult, float]:
    # Warmup
    print("Warming up JIT...")
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
    # Ensure computation is done before stopping timer
    trace.block_until_ready()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # trace is jax array, convert to numpy
    # Trace shape is (N_blocks + 1, 3)
    trace_np = np.array(trace)
    a1_sparse = trace_np[:, 0]
    a2_sparse = trace_np[:, 1]
    a3_sparse = trace_np[:, 2]

    # Reconstruct sparse z coordinates
    z_full = np.array(struct.z_coords)
    z_indices = list(range(0, len(z_full), struct.block_size))
    # If the last block is padded/checked, ensure we get the end point
    if z_indices[-1] != len(z_full) - 1:
        z_indices.append(len(z_full) - 1)

    # Adjust if length mismatch (same logic as original)
    if len(z_indices) != len(a1_sparse):
        print(f"Warning: Trace length {len(a1_sparse)} vs Indices length {len(z_indices)}. adjusting...")
        z_sparse = z_full[z_indices[: len(a1_sparse)]]
    else:
        z_sparse = z_full[z_indices]

    # Calculate power for sparse result
    total_power = np.abs(a1_sparse) ** 2 + np.abs(a2_sparse) ** 2 + np.abs(a3_sparse) ** 2

    # Return sparse result WITHOUT interpolation
    return SimulationResult(
        z=z_sparse,
        a1=a1_sparse,
        a2=a2_sparse,
        a3=a3_sparse,
        total_power=total_power
    ), elapsed_time


@memory.cache
def _solve_ode_core(
    z_coords: np.ndarray,
    kappa_shg_vals: np.ndarray,
    kappa_sfg_vals: np.ndarray,
    dk_shg: float,
    dk_sfg: float,
    y0: np.ndarray,
    method: str,
    rtol: float | None = None,
    atol: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    def odes(z: float, A: Tuple[float, float, float]) -> list[Any]:
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


def run_dop853(struct: SimulationStructure) -> Tuple[SimulationResult, float]:
    z_coords = np.array(struct.z_coords)
    kappa_shg_vals = np.array(struct.kappa_shg_vals)
    kappa_sfg_vals = np.array(struct.kappa_sfg_vals)
    dk_shg = float(struct.dk_shg)
    dk_sfg = float(struct.dk_sfg)

    y0 = np.array(struct.p_in, dtype=np.complex128)

    # Hardcoded method and tolerances for verification
    method = "DOP853"
    rtol = 1e-8
    atol = 1e-8

    start_time = time.perf_counter()
    t, y, nfev = _solve_ode_core(z_coords, kappa_shg_vals, kappa_sfg_vals, dk_shg, dk_sfg, y0, method, rtol, atol)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    a1 = y[0]
    a2 = y[1]
    a3 = y[2]
    print(f"DOP853 nfev: {nfev}")

    total_power = np.abs(a1) ** 2 + np.abs(a2) ** 2 + np.abs(a3) ** 2

    return SimulationResult(z=t, a1=a1, a2=a2, a3=a3, total_power=total_power), elapsed_time
