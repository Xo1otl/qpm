from collections import OrderedDict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from femwell.maxwell.waveguide import Mode, compute_modes
from femwell.mesh import mesh_from_OrderedDict
from shapely.geometry import box
from skfem import Basis, ElementTriP1, Mesh
from skfem.io import from_meshio


@dataclass
class SimulationConfig:
    """Configuration parameters (Value Object)."""

    wavelength_um: float = 1.031
    width_min: float = -10.0
    width_max: float = 10.0
    depth_min: float = -2.0
    depth_max: float = 10.0
    core_resolution: float = 0.25
    core_distance: float = 2.0
    num_modes: int = 3
    n_sub: float = 2.1325
    n_guess_offset: float = 0.005


@dataclass
class SimulationContext:
    mesh: Mesh
    basis: Basis
    config: SimulationConfig
    n_dist: np.ndarray


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
    # 1. Define Geometry
    core_zone = box(-3, 1, 3, 5)
    domain = box(cfg.width_min, cfg.depth_min, cfg.width_max, cfg.depth_max)

    shapes = OrderedDict([("core_zone", core_zone), ("domain", domain)])
    resolutions = {
        "core_zone": {"resolution": cfg.core_resolution, "distance": 0},
        "domain": {"resolution": 1.0, "distance": 1},
    }

    # 2. Mesh Generation
    mesh_obj = from_meshio(mesh_from_OrderedDict(shapes, resolutions=resolutions)).with_boundaries({})
    basis_obj = Basis(mesh_obj, ElementTriP1())

    # 3. Physics Initialization (Pre-compute Index Profile)
    # [0] is x (width), [1] is y (depth)
    width_coords = basis_obj.doflocs[0]
    depth_coords = basis_obj.doflocs[1]

    # Gaussian Profile Definition
    delta_n = 0.012
    sigma_width = 1.5
    sigma_depth = 1.5
    center_depth = 3.0

    gaussian = np.exp(-(width_coords**2) / (sigma_width**2) - ((depth_coords - center_depth) ** 2) / (sigma_depth**2))
    n_dist = cfg.n_sub + delta_n * gaussian

    # Clip air region
    n_dist = np.where(depth_coords < 0, 1.0, n_dist)

    return SimulationContext(mesh=mesh_obj, basis=basis_obj, config=cfg, n_dist=n_dist)


def solve_eigenmodes(ctx: SimulationContext) -> ModeList:
    print("Solver: Calculating modes...")

    # Calculate Epsilon from pre-computed n_dist
    epsilon = ctx.n_dist**2

    raw_modes = compute_modes(
        ctx.basis,
        epsilon,
        wavelength=ctx.config.wavelength_um,
        num_modes=ctx.config.num_modes,
        order=1,
        n_guess=ctx.config.n_sub + ctx.config.n_guess_offset,
    )

    print(f"Solver: Found {len(raw_modes)} potential modes.")

    results: ModeList = []
    for i, mode in enumerate(raw_modes):  # pyright: ignore[reportArgumentType]
        n_eff = float(mode.n_eff.real)
        # Guided condition: Effective index > Substrate index
        is_guided = n_eff > ctx.config.n_sub
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
    guided_count = sum(1 for m in results if m.is_guided)

    if guided_count == 0:
        print("FAIL: No modes guided.")
    elif guided_count <= 2:
        print("SUCCESS: Single-Mode operation achieved (Fundamental Pair only).")
    else:
        print("FAIL: Higher Order Modes (HOMs) are still guided.")


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
        vmin=ctx.config.n_sub,
        vmax=n_core_max,
    )

    mappable = axes[0].collections[-1]
    plt.colorbar(mappable, ax=axes[0], label="n")
    axes[0].set_title(f"Refractive Index (Clamped to {ctx.config.n_sub:.3f}+)")
    axes[0].set_aspect("equal")

    # --- Plot B: Mode Intensity ---
    result.field_data.plot_intensity(ax=axes[1], colorbar=True)
    axes[1].set_title(f"Mode {result.index} Intensity (n_eff={result.n_eff:.5f})")
    axes[1].set_aspect("equal")
    axes[1].set_ylabel("")

    plt.tight_layout()
    filename = f"manual_mode_{result.index}.png"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()


def run() -> None:
    print("--- RUNNING SINGLE-MODE MOCK SIMULATION ---")

    # 1. Initialization
    cfg = SimulationConfig()
    ctx = new_simulation_context(cfg)

    # 2. Computation
    modes = solve_eigenmodes(ctx)

    # 3. Visualization
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
