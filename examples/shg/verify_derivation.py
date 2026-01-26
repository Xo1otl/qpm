import numpy as np


def verify_derivation():
    # Parameters
    L = 1000.0  # Total length (microns)
    Lambda0 = 10.0  # Fundamental period (microns)
    m = 1  # QPM order
    Gm = 2 * np.pi * m / Lambda0

    # Mismatch
    # Let's say we have a small phase mismatch delta_beta
    # delta_k = Gm + delta_beta
    delta_beta = 0.01  # inverse microns
    delta_k = Gm + delta_beta

    # Discretization
    N = int(L / Lambda0)

    # Profiles
    # Let's use a slowly varying duty cycle and phase shift
    def D(z):
        return 0.5 + 0.2 * np.sin(2 * np.pi * z / L)

    def delta(z):
        # delta_n in the derivation
        return 1.0 * np.cos(2 * np.pi * z / L)

    # 1. Exact Discrete Sum
    # A2 ~ sum_{n=0}^{N-1} e^{i delta_k z_n} * I_unit(n)
    # where I_unit(n) approx -Lambda0 * (2/m*pi) * sin(m*pi*Dn) * e^{i Gm delta_n}
    # (Using the approx I_unit from the doc which is verified as correct unit cell integral)

    discrete_sum = 0j
    for n in range(N):
        zn = n * Lambda0 + Lambda0 / 2
        Dn = D(zn)
        dn = delta(zn)
        phi_n = Gm * dn

        # The unit cell integral (verified correct in both docs)
        I_unit = -Lambda0 * (2 / (m * np.pi)) * np.sin(m * np.pi * Dn) * np.exp(1j * phi_n)

        term = np.exp(1j * delta_k * zn) * I_unit
        discrete_sum += term

    print(f"Discrete Sum (Truth): {discrete_sum:.6g}")

    # 2. Old Continuum Limit (from min.md)
    # Integral of [ term(z) ] * e^{i delta_k z} dz
    # This ignores aliasing and tries to integrate high frequency e^{i delta_k z}
    # against a sampled envelope, essentially.
    # The doc formula:
    # int_0^L [ -2/(m*pi) * sin(m*pi*D(z)) * e^{i Gm delta(z)} ] * e^{i delta_k z} dz

    # We compute this numerically with high resolution
    z_dense = np.linspace(0, L, 10000)
    dz = z_dense[1] - z_dense[0]

    old_integrand = -(2 / (m * np.pi)) * np.sin(m * np.pi * D(z_dense)) * np.exp(1j * Gm * delta(z_dense)) * np.exp(1j * delta_k * z_dense)
    old_integral = np.sum(old_integrand) * dz

    print(f"Old Integral (min.md): {old_integral:.6g}")

    # 3. New Continuum Limit (from fix.md)
    # Integral of [ term(z) * (-1)^m ] * e^{i delta_beta z} dz
    # effectively demodulating the carrier

    # Note: (-1)^m * (-2/(m*pi)) = (-1)^{m+1} * 2/(m*pi)
    prefactor = (-1) ** (m + 1) * (2 / (m * np.pi))

    new_integrand = prefactor * np.sin(m * np.pi * D(z_dense)) * np.exp(1j * Gm * delta(z_dense)) * np.exp(1j * delta_beta * z_dense)
    new_integral = np.sum(new_integrand) * dz

    print(f"New Integral (fix.md): {new_integral:.6g}")

    # Comparisons
    err_old = abs(old_integral - discrete_sum) / abs(discrete_sum)
    err_new = abs(new_integral - discrete_sum) / abs(discrete_sum)

    print(f"\nRelative Error Old: {err_old:.4%}")
    print(f"Relative Error New: {err_new:.4%}")

    if err_new < 0.01 and err_old > 0.5:
        print("\nSUCCESS: New formulation matches discrete sum, old one fails.")
    else:
        print("\nFAILURE: Unexpected results.")


if __name__ == "__main__":
    verify_derivation()
