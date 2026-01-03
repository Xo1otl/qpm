import jax.numpy as jnp

from qpm import cwes2, mgoslt


def test_cwes_consistency() -> None:
    """
    Verifies consistency between the full perturbation calculation (simulate_twm)
    and the analytical NPDA approximation (calc_s_analytical) when kappa_SHG = kappa_SFG.
    """
    # --- Setup Physical Constants and Design Parameters ---
    # Same parameters as test_npda.py
    design_wl = 1.031
    design_temp = 70.0
    num_domains_shg = 321
    num_domains_sfg = 1168
    kappa_mag = 1.31e-5 / (2 / jnp.pi)

    # --- Calculate Phase Mismatches ---
    dk1 = mgoslt.calc_twm_delta_k(design_wl, design_wl, design_temp)
    dk2 = mgoslt.calc_twm_delta_k(design_wl, design_wl / 2, design_temp)

    # --- Define QPM Grating Structure ---
    shg_width = jnp.pi / dk1
    sfg_width = jnp.pi / dk2
    widths_shg = jnp.array([shg_width] * num_domains_shg)
    widths_sfg = jnp.array([sfg_width] * num_domains_sfg)
    widths = jnp.concatenate([widths_shg, widths_sfg])

    # Kappas alternate in sign for QPM
    num_domains = num_domains_shg + num_domains_sfg
    kappas = kappa_mag * (-1) ** jnp.arange(num_domains)

    # --- 1. Analytical Calculation (NPDA) ---
    a3_npda_mag = jnp.abs(cwes2.calc_a3_npda(1, kappas, kappas, widths, dk1, dk2))

    # --- 2. Perturbation Calculation (Full CWE) ---
    b0 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex64)
    # Pass kappas for both SHG and SFG
    b_final = cwes2.simulate_twm(widths, kappas, kappas, dk1, dk2, b0)

    # |A3| = |B3| (phase rotation preserves magnitude)
    a3_cwes_mag = jnp.abs(b_final[2])

    # --- Verification ---
    # Comparison: NPDA is an approximation (undepleted pump), while simulate_twm includes pump depletion (though small here).
    # We expect them to be reasonably close (e.g., within 1%).
    print(f"NPDA |A3|: {a3_npda_mag}")
    print(f"CWES |A3|: {a3_cwes_mag}")

    assert jnp.allclose(a3_cwes_mag, a3_npda_mag, rtol=1e-2)
