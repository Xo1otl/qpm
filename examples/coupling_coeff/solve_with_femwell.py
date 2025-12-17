from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from calculate_index import ProcessParams, concentration_distribution, get_delta_n0, new_standard_process_params
from femwell.maxwell.waveguide import Modes, compute_modes
from femwell.mesh import mesh_from_OrderedDict
from shapely.geometry import box
from skfem import Basis, ElementTriP1, MeshTri
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


def get_refractive_index_function(params: ProcessParams, wavelength_um: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    n_sub = mgoslt.sellmeier_n_eff(wavelength_um, params.temp_c)
    delta_n0 = np.array(get_delta_n0(wavelength_um))

    def refractive_index_fun(depth: np.ndarray, width: np.ndarray) -> np.ndarray:
        c_norm = np.array(concentration_distribution(jnp.array(depth), jnp.array(width), params))
        n_dist = n_sub + delta_n0 * c_norm
        # Apply air interface condition (Air for depth < 0)
        return np.where(depth < 0, 1.0, n_dist)

    return refractive_index_fun


def create_simulation_mesh(config: SimulationConfig) -> MeshTri:
    """Creates the finite element mesh for the simulation."""
    domain = box(config.width_min, config.depth_min, config.width_max, config.depth_max)

    shapes = OrderedDict([("core", domain)])
    resolutions = {
        "core": {
            "resolution": config.core_resolution,
            "distance": config.core_distance,
        },
    }

    meshio_mesh = mesh_from_OrderedDict(shapes, resolutions=resolutions)
    return from_meshio(meshio_mesh).with_boundaries({})


def solve_modes(config: SimulationConfig) -> tuple[Modes, float]:
    """
    Solves for the optical modes of the waveguide.

    Args:
        config: Simulation configuration.

    Returns:
        Tuple containing:
        - List of mode objects found by the solver.
        - Substrate refractive index (n_sub).
    """
    params = new_standard_process_params()
    mesh = create_simulation_mesh(config)
    basis = Basis(mesh, ElementTriP1())

    n_sub = mgoslt.sellmeier_n_eff(config.wavelength_um, params.temp_c)

    n_fun = get_refractive_index_function(params, config.wavelength_um)

    # Project permittivity (epsilon = n^2) onto the basis
    # basis.doflocs is [x, y] -> [width, depth]
    width_vals = basis.doflocs[0]
    depth_vals = basis.doflocs[1]

    n_vals = n_fun(depth_vals, width_vals)
    epsilon = n_vals**2

    peak_n = np.sqrt(epsilon.max())

    modes = compute_modes(basis, epsilon, wavelength=config.wavelength_um, num_modes=config.num_modes, order=1, n_guess=peak_n)
    return modes, n_sub


def run() -> None:
    config = SimulationConfig()
    modes, n_sub = solve_modes(config)

    print(f"Found {len(modes)} modes.")
    print(f"Substrate Index (n_sub): {n_sub:.6f}")

    if not modes:
        print("No modes found.")
        return

    mode0_neff = modes[0].n_eff.real
    print(f"Mode 0 (Fundamental): n_eff = {mode0_neff:.6f}")

    if mode0_neff <= n_sub:
        print(f"Fundamental mode is not guided! n_eff ({mode0_neff:.6f}) <= n_sub ({n_sub:.6f})")

    if len(modes) > 1:
        mode1_neff = modes[1].n_eff.real
        print(f"Mode 1: n_eff = {mode1_neff:.6f}")
        if mode1_neff > n_sub:
            print(f"Waveguide is NOT single-mode! Mode 1 is guided: n_eff ({mode1_neff:.6f}) > n_sub ({n_sub:.6f})")

    if config.plot_modes:
        plt.figure()
        modes[0].plot(np.real(modes[0].E), plot_vectors=True, colorbar=True)
        plt.title(f"Mode 0 (TE00) n={modes[0].n_eff.real:.4f}")
        plt.xlabel("Width (um)")
        plt.ylabel("Depth (um)")
        plt.savefig("mode_0.png")
        print("Saved mode_0.png")


if __name__ == "__main__":
    run()
