import jax
import jax.numpy as jnp


def calculate_gdd(wls_um: jax.Array, amps: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    # c in um / ps
    c = 299.792458
    # omega in rad / ps
    omegas = 2 * jnp.pi * c / wls_um

    # Sort by omega ascending
    idx_sorted = jnp.argsort(omegas)
    omegas_sorted = omegas[idx_sorted]
    amps_sorted = amps[idx_sorted]

    phi = jnp.unwrap(jnp.angle(amps_sorted))

    d_omega = jnp.diff(omegas_sorted)

    # GD
    d_phi = jnp.diff(phi)
    gd_mid = d_phi / d_omega
    omega_mid = (omegas_sorted[1:] + omegas_sorted[:-1]) / 2.0

    # GDD
    d_omega_mid = jnp.diff(omega_mid)
    d_gd = jnp.diff(gd_mid)
    gdd_mid = d_gd / d_omega_mid
    omega_mid2 = (omega_mid[1:] + omega_mid[:-1]) / 2.0

    return omega_mid2, gd_mid, gdd_mid


def test_gdd() -> None:
    # Grid: 1000 points over 2nm is ~3.5 rad/ps bandwidth
    wls = jnp.linspace(1.030, 1.032, 1000)
    c = 299.792458
    omegas = 2 * jnp.pi * c / wls
    center_omega = 2 * jnp.pi * c / 1.031

    # Make pulse width reasonable relative to grid
    # Sigma ~ 0.5 rad/ps
    sigma_w = 0.5

    # Case 1: Transform Limited (Linear Phase)
    amp1 = jnp.exp(-((omegas - center_omega) ** 2) / (2 * sigma_w**2))
    phase1 = 10.0 * (omegas - center_omega)
    field1 = amp1 * jnp.exp(1j * phase1)

    # Note: mask is on original omegas. We need to mask the outputs,
    # but calculating GDD on noise is fine as long as we check the center.

    _, gd1, gdd1 = calculate_gdd(wls, field1)

    # Interpolate amp back to o_grid to weight/mask
    # Or just check central index
    center_idx = len(gdd1) // 2

    print("Case 1 (TL)")
    print(f"  Center GD: {gd1[center_idx]:.4f} ps (Expected 10.0)")
    print(f"  Center GDD: {gdd1[center_idx]:.6e} ps^2 (Expected ~0)")

    # Case 2: Chirped (Quadratic Phase)
    beta2 = 1.0  # ps^2
    phase2 = 0.5 * beta2 * (omegas - center_omega) ** 2
    # Add cubic phase to test? No keep simple.
    field2 = amp1 * jnp.exp(1j * phase2)

    _, _, gdd2 = calculate_gdd(wls, field2)

    print("Case 2 (Chirped)")
    print(f"  Center GDD: {gdd2[center_idx]:.6e} ps^2 (Expected ~1.0)")


if __name__ == "__main__":
    test_gdd()
