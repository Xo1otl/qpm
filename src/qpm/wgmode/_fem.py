from collections import OrderedDict
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from femwell.maxwell.waveguide import Mode, compute_modes
from femwell.mesh import mesh_from_OrderedDict
from joblib import Memory
from shapely.geometry import box
from skfem import Basis, ElementTriP1, Mesh
from skfem.io import from_meshio

from qpm import ape, mgoslt

memory = Memory(location=".cache", verbose=0)


@dataclass
class SimulationConfig:
    """Configuration for waveguide simulation."""

    wavelength_um: float
    width_min: float
    width_max: float
    depth_min: float
    depth_max: float
    core_resolution: float
    cladding_resolution: float
    core_width_half: float
    core_depth_max: float
    core_distance: float
    num_modes: int
    plot_modes: bool
    n_guess_offset: float
    process_params: ape.ProcessParams
    upper_cladding_n: float
    apply_upper_cladding: bool | None


@dataclass
class SimulationContext:
    """Context containing mesh and physics data for simulation."""

    mesh: Mesh
    basis: Basis
    config: SimulationConfig
    n_dist: np.ndarray
    n_sub: float


@dataclass
class ModeResult:
    """Result of a mode calculation."""

    index: int
    n_eff: float
    field_data: Mode
    is_guided: bool


type ModeList = list[ModeResult]


# NOTE: The code is too tightly coupled; individual processes should be separeted into functions as defined in the documentation rather than being hardcoded.
def new_simulation_context(cfg: SimulationConfig) -> SimulationContext:
    """
    Creates a new simulation context with mesh and refractive index distribution.
    """
    # 1. Define Geometry and Mesh
    full_domain = box(cfg.width_min, cfg.depth_min, cfg.width_max, cfg.depth_max)
    core_domain = box(-cfg.core_width_half, cfg.depth_min, cfg.core_width_half, cfg.core_depth_max)
    cladding_domain = full_domain.difference(core_domain)

    shapes = OrderedDict([("core", core_domain), ("cladding", cladding_domain)])
    resolutions = {
        "core": {
            "resolution": cfg.core_resolution,
            "distance": cfg.core_distance,
        },
        "cladding": {
            "resolution": cfg.cladding_resolution,  # Coarser resolution
            "distance": cfg.core_distance,
        },
    }

    mesh_obj = from_meshio(mesh_from_OrderedDict(shapes, resolutions=resolutions)).with_boundaries({})
    basis_obj = Basis(mesh_obj, ElementTriP1())

    # 2. Physics Initialization (Pre-compute Index Profile)
    params = cfg.process_params

    # Calculate substrate index
    n_sub = mgoslt.sellmeier_n_eff(cfg.wavelength_um, params.temp_c)

    # Calculate index distribution
    delta_n0 = np.array(ape.get_delta_n0(cfg.wavelength_um))

    # basis.doflocs is [x, y] -> [width, depth]
    width_vals = basis_obj.doflocs[0]
    depth_vals = basis_obj.doflocs[1]

    # Use the diffusion model from calculate_index
    c_norm = np.array(ape.concentration_distribution(jnp.array(depth_vals), jnp.array(width_vals), params))
    n_dist = n_sub + delta_n0 * c_norm

    # Apply upper cladding condition
    should_apply_cladding = cfg.apply_upper_cladding
    if should_apply_cladding is None:
        should_apply_cladding = not params.is_buried

    if should_apply_cladding:
        n_dist = np.where(depth_vals < 0, cfg.upper_cladding_n, n_dist)

    # Ensure n_dist is a numpy array (not JAX) to avoid issues with scipy
    n_dist = np.array(n_dist)

    return SimulationContext(mesh=mesh_obj, basis=basis_obj, config=cfg, n_dist=n_dist, n_sub=n_sub)


def solve_eigenmodes(ctx: SimulationContext) -> ModeList:
    """Solves for eigenmodes using the provided context."""
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
    """Finds the fundamental TM00 mode from the results."""
    # Filter for guided TM modes
    tm_modes = [m for m in results if m.is_guided and m.field_data.te_fraction < 0.5]

    if not tm_modes:
        return None

    # Sort by n_eff descending
    tm_modes.sort(key=lambda m: m.n_eff, reverse=True)
    return tm_modes[0]


@memory.cache
def compute_modes_from_config(cfg: SimulationConfig) -> tuple[SimulationContext, ModeList]:
    """
    Wrapper for solve_eigenmodes that caches results based on the configuration.
    This avoids re-running the expensive FEM solver if the config hasn't changed.
    """
    ctx = new_simulation_context(cfg)
    return ctx, solve_eigenmodes(ctx)


def compute_tm00(cfg: SimulationConfig) -> ModeResult | None:
    """
    Finds the fundamental TM00 mode for the given configuration.
    Uses cached compute_modes_from_config.
    """
    _, modes = compute_modes_from_config(cfg)
    return find_tm00_mode(modes)
