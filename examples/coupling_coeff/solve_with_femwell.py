from collections import OrderedDict
from dataclasses import dataclass

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from calculate_index import concentration_distribution, get_delta_n0, new_standard_process_params
from femwell.maxwell.waveguide import Mode, compute_modes
from femwell.mesh import mesh_from_OrderedDict
from shapely.geometry import box
from skfem import Basis, ElementTriP1, Mesh
from skfem.io import from_meshio

from qpm import mgoslt


@dataclass
class SimulationConfig:
    """Configuration for the waveguide mode simulation."""

    wavelength_um: float = 1.031
    width_min: float = -60.0
    width_max: float = 60.0
    depth_min: float = -5.0
    depth_max: float = 25.0
    core_resolution: float = 0.5
    core_distance: float = 2.0
    num_modes: int = 3
    plot_modes: bool = True
    n_guess_offset: float = 5e-3


@dataclass
class SimulationContext:
    mesh: Mesh
    basis: Basis
    config: SimulationConfig
    n_dist: np.ndarray
    n_sub: float


@dataclass
class ModeResult:
    index: int
    n_eff: float
    field_data: Mode
    is_guided: bool


type ModeList = list[ModeResult]


def new_simulation_context(cfg: SimulationConfig) -> SimulationContext:
    """
    Factory: centralized initialization logic.
    Constructs the mesh, basis, and pre-calculates the material distribution.
    """
    # 1. Define Geometry and Mesh
    domain = box(cfg.width_min, cfg.depth_min, cfg.width_max, cfg.depth_max)
    shapes = OrderedDict([("core", domain)])
    resolutions = {
        "core": {
            "resolution": cfg.core_resolution,
            "distance": cfg.core_distance,
        },
    }

    mesh_obj = from_meshio(mesh_from_OrderedDict(shapes, resolutions=resolutions)).with_boundaries({})
    basis_obj = Basis(mesh_obj, ElementTriP1())

    # 2. Physics Initialization (Pre-compute Index Profile)
    params = new_standard_process_params()

    # Calculate substrate index
    n_sub = mgoslt.sellmeier_n_eff(cfg.wavelength_um, params.temp_c)

    # Calculate index distribution
    delta_n0 = np.array(get_delta_n0(cfg.wavelength_um))

    # basis.doflocs is [x, y] -> [width, depth]
    width_vals = basis_obj.doflocs[0]
    depth_vals = basis_obj.doflocs[1]

    # Use the diffusion model from calculate_index
    c_norm = np.array(concentration_distribution(jnp.array(depth_vals), jnp.array(width_vals), params))
    n_dist = n_sub + delta_n0 * c_norm

    # Apply air interface condition (Air for depth < 0)
    n_dist = np.where(depth_vals < 0, 1.0, n_dist)

    return SimulationContext(mesh=mesh_obj, basis=basis_obj, config=cfg, n_dist=n_dist, n_sub=n_sub)


def solve_eigenmodes(ctx: SimulationContext) -> ModeList:
    print("Solver: Calculating modes...")

    # Calculate Epsilon from pre-computed n_dist
    epsilon = ctx.n_dist**2

    # Use peak index for initial guess
    peak_n = np.max(ctx.n_dist)

    raw_modes = compute_modes(
        ctx.basis,
        epsilon,
        wavelength=ctx.config.wavelength_um,
        num_modes=ctx.config.num_modes,
        order=1,
        n_guess=peak_n,
    )

    print(f"Solver: Found {len(raw_modes)} potential modes.")

    results: ModeList = []
    for i, mode in enumerate(raw_modes):  # pyright: ignore[reportArgumentType]
        n_eff = float(mode.n_eff.real)
        # Guided condition: Effective index > Substrate index
        is_guided = n_eff > ctx.n_sub
        results.append(ModeResult(i, n_eff, mode, is_guided))

    return results


def find_tm00_mode(results: ModeList) -> ModeResult | None:
    # Filter for guided TM modes
    tm_modes = [m for m in results if m.is_guided and m.field_data.te_fraction < 0.5]

    if not tm_modes:
        return None

    # Sort by n_eff descending
    tm_modes.sort(key=lambda m: m.n_eff, reverse=True)
    return tm_modes[0]


def verify_single_mode_condition(results: ModeList) -> None:
    guided_modes = [m for m in results if m.is_guided]
    guided_count = len(guided_modes)

    print(f"Verification: Found {guided_count} guided modes.")

    if guided_count == 0:
        print("FAIL: No modes guided.")
    elif guided_count == 1:
        print("SUCCESS: Single-Mode operation achieved (Fundamental only).")
    else:
        # Check if it's strictly single mode (only mode 0) or if we allow degenerate pairs (like TE/TM)
        # For this specific task, usually we want just one guided mode or we check specifically for higher order.
        print(f"FAIL: Multi-mode behavior detected ({guided_count} modes guided).")
        for m in guided_modes:
            print(f"  - Mode {m.index}: n_eff={m.n_eff:.6f}")


def plot_mode_result(ctx: SimulationContext, result: ModeResult) -> None:
    if not result.is_guided:
        print(f"Mode {result.index}: n_eff = {result.n_eff:.6f} (Cutoff - Ignored)")
        return

    print(f"Mode {result.index}: n_eff = {result.n_eff:.6f} (Guided)")

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot A: Refractive Index ---
    n_core_max = np.max(ctx.n_dist)

    ctx.basis.plot(
        ctx.n_dist,
        ax=axes[0],
        cmap="viridis",
        shading="gouraud",
        # Clamp visualization to better see core contrast
        vmin=ctx.n_sub,
        vmax=n_core_max,
    )

    mappable = axes[0].collections[-1]
    plt.colorbar(mappable, ax=axes[0], label="n")
    axes[0].set_title(f"Refractive Index (Clamped to {ctx.n_sub:.3f}+)")
    axes[0].set_aspect("equal")

    # --- Plot B: Mode Intensity ---
    result.field_data.plot_intensity(ax=axes[1], colorbar=True)
    axes[1].set_title(f"Mode {result.index} Intensity (n_eff={result.n_eff:.5f})")
    axes[1].set_aspect("equal")
    axes[1].set_ylabel("")

    plt.tight_layout()
    filename = f"sim_mode_{result.index}.png"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()


def run() -> None:
    print("--- RUNNING WAVEGUIDE SIMULATION ---")

    # 1. Initialization
    cfg = SimulationConfig()
    ctx = new_simulation_context(cfg)

    print(f"Substrate Index (n_sub): {ctx.n_sub:.6f}")

    # 2. Computation
    modes = solve_eigenmodes(ctx)

    # 3. Visualization
    if cfg.plot_modes:
        tm00 = find_tm00_mode(modes)
        if tm00:
            print(f"Plotting TM00 Mode (Mode Index {tm00.index})")
            plot_mode_result(ctx, tm00)
        else:
            print("Warning: No TM00 mode found to plot.")

    # 4. Verification
    verify_single_mode_condition(modes)


if __name__ == "__main__":
    run()
