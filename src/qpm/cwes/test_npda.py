import jax
import jax.numpy as jnp

from qpm import cwes, mgoslt


# --- Parameter Conversion ---
def conv_params_to_simplified(kappas: jax.Array, widths: jax.Array) -> tuple[jax.Array, float]:
    """
    Converts piecewise-constant parameters (kappas, widths) to a uniformly
    sampled kappa array and dz for the simplified Riemann sum.
    """
    # Hardcode dz to a small value for a fine grid, as requested
    dz = 0.0001

    # Get domain end points [z_1, z_2, ..., z_N]
    z_end_points = jnp.cumsum(widths)
    length = z_end_points[-1]

    # Create a uniform z grid from 0 to L
    # Use ceil to ensure the grid covers the entire length L
    num_points = jnp.ceil(length / dz).astype(int)
    # z_grid represents the start of each dz interval: [0, dz, 2*dz, ...]
    z_grid = jnp.arange(num_points) * dz

    # Find which domain each z_grid point falls into
    # jnp.searchsorted finds the index 'i' such that z_end_points[i-1] < z <= z_end_points[i]
    # This maps each z_grid point to its corresponding kappa index.
    indices = jnp.searchsorted(z_end_points, z_grid, side="right")

    # Create the simplified kappa array by indexing
    kappas_simplified = kappas[indices]

    return kappas_simplified, dz


# --- Simplified Implementation (for context) ---
@jax.jit
def calc_s_simplified(kappas: jax.Array, dz: float, dk1: jax.Array, dk2: jax.Array) -> jnp.ndarray:
    """
    Calculates the S-functional using a simplified Riemann sum on a uniform grid.
    """
    n = kappas.shape[0]
    z = jnp.arange(n) * dz
    v = kappas * jnp.exp(-1j * dk1 * z)
    b_naive = jnp.cumsum(v) * dz
    # Pad and shift to get the sum up to (n-1)
    b = jnp.pad(b_naive, (1, 0))[:-1]
    u = kappas * jnp.exp(-1j * dk2 * z)
    return jnp.sum(u * b) * dz


def test_npda() -> None:
    # --- Setup Physical Constants and Design Parameters ---
    design_wl = 1.031
    design_temp = 70.0
    num_domains_shg = 321
    num_domains_sfg = 1168
    kappa_mag = 1.31e-5 / (2 / jnp.pi)

    # --- Calculate Phase Mismatches ---
    # Base phase mismatches for the design parameters
    dk1 = mgoslt.calc_twm_delta_k(design_wl, design_wl, design_temp)
    dk2 = mgoslt.calc_twm_delta_k(design_wl, design_wl / 2, design_temp)

    # --- Define QPM Grating Structure (kappas and widths) ---
    # Widths are set for perfect phase matching at base dk values
    shg_width = jnp.pi / dk1
    sfg_width = jnp.pi / dk2
    widths_shg = jnp.array([shg_width] * num_domains_shg)
    widths_sfg = jnp.array([sfg_width] * num_domains_sfg)
    widths = jnp.concatenate([widths_shg, widths_sfg])

    # Kappas alternate in sign for QPM
    num_domains = num_domains_shg + num_domains_sfg
    kappas = kappa_mag * (-1) ** jnp.arange(num_domains)

    # --- Execute and Compare Calculation Methods ---
    # 1. Analytical solution
    s_analytical = cwes.calc_s_analytical(kappas, widths, dk1, dk2)

    # 2. Simplified direct numerical approximation
    kappas_simplified, dz = conv_params_to_simplified(kappas, widths)
    s_simplified = calc_s_simplified(kappas_simplified, dz, dk1, dk2)

    # --- Verification ---
    # Compare the absolute values of the complex results.
    assert jnp.allclose(jnp.abs(s_analytical), jnp.abs(s_simplified), rtol=1e-8)
