from collections import OrderedDict
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from femwell.mesh import mesh_from_OrderedDict
from scipy.optimize import root_scalar
from shapely.geometry import box
from skfem import Basis, ElementTriP1
from skfem.io import from_meshio

from qpm import ape, wgmode


def solve_slab_analytical_modes(n_core: float, n_clad: float, thickness_um: float, wavelength_um: float) -> tuple[float, float]:
    """
    Solves for the fundamental TE0 and TM0 mode effective indices.
    """
    k0 = 2 * np.pi / wavelength_um
    d = thickness_um
    V = k0 * (d / 2) * np.sqrt(n_core**2 - n_clad**2)
    upper_limit = min(V, np.pi / 2 - 1e-6)

    # TE Solver
    def func_u_te(u: float) -> float:
        lhs = u * np.tan(u)
        rhs = np.sqrt(V**2 - u**2)
        return lhs - rhs

    # TM Solver
    def func_u_tm(u: float) -> float:
        lhs = u * np.tan(u)
        rhs = (n_core / n_clad) ** 2 * np.sqrt(V**2 - u**2)
        return lhs - rhs

    def get_neff(func: Callable[[float], float]) -> float:
        try:
            sol = root_scalar(func, bracket=[1e-6, upper_limit], method="brentq")
            u = sol.root
            val = (u / (k0 * d / 2)) ** 2
            return np.sqrt(n_core**2 - val)
        except ValueError:
            return n_clad

    return get_neff(func_u_te), get_neff(func_u_tm)


def test_fem_vs_analytical_slab() -> None:
    """
    Compares FEM solver against analytical solution for a symmetric slab waveguide.
    """
    # Parameters
    wavelength_um = 1.55
    n_core = 2.1
    n_clad = 1.444
    core_thickness = 1.0

    # 1. Analytical Solution
    n_eff_te, n_eff_tm = solve_slab_analytical_modes(n_core, n_clad, core_thickness, wavelength_um)
    print(f"Analytical n_eff: TE0={n_eff_te:.6f}, TM0={n_eff_tm:.6f}")

    # 2. FEM Setup
    # Create fake config (mostly ignored since we overwrite ctx)
    fake_params = ape.ProcessParams(
        temp_c=100.0,
        d_pe_coeff=0.1,
        t_pe_hours=1.0,
        mask_width_um=1.0,
        t_anneal_hours=1.0,
        d_x_coeff=0.1,
        d_y_coeff=0.1,
        is_buried=False,
    )

    # Domain
    x_min, x_max = -10.0, 10.0
    y_min, y_max = -5.0, 5.0
    core_w = 40.0
    core_h = core_thickness

    full_domain = box(x_min, y_min, x_max, y_max)
    core_domain = box(x_min, -core_h / 2, x_max, core_h / 2)
    cladding_domain = full_domain.difference(core_domain)

    shapes = OrderedDict([("core", core_domain), ("cladding", cladding_domain)])
    resolutions = {
        "core": {"resolution": 0.05, "distance": 0.5},
        "cladding": {"resolution": 0.2, "distance": 0.5},
    }

    mesh_obj = from_meshio(mesh_from_OrderedDict(shapes, resolutions=resolutions)).with_boundaries({})
    basis_obj = Basis(mesh_obj, ElementTriP1())

    depth_vals = basis_obj.doflocs[1]
    n_dist = np.ones_like(depth_vals) * n_clad
    mask_core = (depth_vals >= -core_h / 2) & (depth_vals <= core_h / 2)
    n_dist[mask_core] = n_core

    config = wgmode.SimulationConfig(
        wavelength_um=wavelength_um,
        width_min=x_min,
        width_max=x_max,
        depth_min=y_min,
        depth_max=y_max,
        core_resolution=0.1,
        cladding_resolution=0.5,
        core_width_half=core_w / 2,
        core_depth_max=core_h / 2,
        core_distance=0.5,
        num_modes=10,
        plot_modes=False,
        n_guess_offset=0.0,
        process_params=fake_params,
        upper_cladding_n=n_clad,
        apply_upper_cladding=True,
        n_sub=n_clad,
        delta_n0=jnp.array([0.0]),
    )

    ctx = wgmode.SimulationContext(mesh=mesh_obj, basis=basis_obj, config=config, n_dist=n_dist, n_sub=n_clad)

    results = wgmode.solve_eigenmodes(ctx)

    print("\nFEM Results:")
    tm_match = False

    for res in results:
        te_frac = res.field_data.te_fraction
        print(f"Mode {res.index}: n_eff={res.n_eff:.6f}, te_fraction={te_frac:.3f}, guided={res.is_guided}")

        # Check against TM (te_fraction < 0.5)
        if te_frac < 0.5:
            if np.isclose(res.n_eff, n_eff_tm, atol=2e-2):
                tm_match = True
                print(f"  -> MATCHED TM0 Analytical ({n_eff_tm:.6f})")
        elif np.isclose(res.n_eff, n_eff_te, atol=2e-2):
            print(f"  -> MATCHED TE0 Analytical ({n_eff_te:.6f})")

    assert tm_match, f"Could not find FEM TM0 mode matching analytical {n_eff_tm:.6f}"
    # assert te_match, f"Could not find FEM TE0 mode matching analytical {n_eff_te:.6f}"


if __name__ == "__main__":
    test_fem_vs_analytical_slab()
