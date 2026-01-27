import numpy as np
from scipy.optimize import fsolve


def inverse_design_gaussian():
    # --- 1. Parameters ---
    L = 1000.0  # Total length (microns)
    Lambda0 = 10.0  # Grating period (microns)
    N = int(L / Lambda0)  # Number of periods
    width = 150.0  # Gaussian width parameter

    # Target Profile: Gaussian Amplitude centered at L/2
    # Normalized so peak is 1.0 (corresponds to G=0.5)
    def target_kappa(z):
        return np.exp(-((z - L / 2) ** 2) / (width**2))

    # --- 2. Method A: Fejer's Continuous Modulation ---
    # G(z) = (1/pi) * arcsin( target(z) )
    # Walls where cos(K0*z) = cos(pi*G(z))
    # Expansion: K0*z = 2*pi*n +/- pi*G(z)
    # z = n*Lambda +/- (Lambda/2)*G(z)

    fejer_walls = []
    # We solve strictly using the transcendental equation to be rigorous
    # Function to find root: cos(2*pi*z/Lambda) - cos(arcsin(target(z))) = 0
    # This simplifies to: cos(2*pi*z/Lambda) - sqrt(1 - target(z)^2) = 0
    # But analytically, the solutions near n*Lambda are roughly n*Lambda +/- (L/2)*G

    # Let's use the exact analytical expansion derived from Fejer's condition:
    # z = n*Lambda +/- (Lambda/2) * G(z)
    # Since G(z) varies slowly, we can approximate G(z) ~ G(n*Lambda) for initialization
    # or solve z - n*Lambda +/- (Lambda/2)*G(z) = 0

    for n in range(N):
        center = (n + 0.5) * Lambda0  # Centering to match typical cos range [0, 2pi]

        # Local Duty Cycle Function
        def duty_func(z):
            val = target_kappa(z)
            # Clip to valid range for arcsin
            val = max(0.0, min(1.0, val))
            return (1 / np.pi) * np.arcsin(val)

        # Solve for Left and Right edges of the domain in this period
        # Initial guess: standard locations
        guess_G = duty_func(center)
        guess_L = center - (Lambda0 / 2) * guess_G
        guess_R = center + (Lambda0 / 2) * guess_G

        # Solving z = center +/- (Lambda/2)*G(z)
        # Note: In Fejer's sign convention, the domain is defined by cos > cos(pi G).
        # This creates a domain centered at n*Lambda (or n*Lambda + 0.5 depending on phase).
        # We align grids by choosing the center to match Method B.

        # Rigorous root finding for Method A
        func_L = lambda z: z - (center - (Lambda0 / 2) * duty_func(z))
        func_R = lambda z: z - (center + (Lambda0 / 2) * duty_func(z))

        root_L = fsolve(func_L, guess_L)[0]
        root_R = fsolve(func_R, guess_R)[0]

        fejer_walls.append((root_L, root_R))

    # --- 3. Method B: Custom Shifted Pulse Model ---
    # Discrete calculation at fixed centers
    custom_walls = []

    for n in range(N):
        # Grid definition: Centers aligned with Method A for valid comparison
        z_n = (n + 0.5) * Lambda0

        # Discrete Amplitude Sample
        A_n = target_kappa(z_n)

        # Discrete Duty Cycle
        D_n = (1 / np.pi) * np.arcsin(A_n)

        # Explicit Structure Construction
        wall_L = z_n - (Lambda0 * D_n / 2)
        wall_R = z_n + (Lambda0 * D_n / 2)

        custom_walls.append((wall_L, wall_R))

    # --- 4. Verification ---
    max_diff = 0.0
    for i in range(N):
        f_L, f_R = fejer_walls[i]
        c_L, c_R = custom_walls[i]

        diff_L = abs(f_L - c_L)
        diff_R = abs(f_R - c_R)
        max_diff = max(max_diff, diff_L, diff_R)

    return max_diff


# Calculate
discrepancy = inverse_design_gaussian()
print(f"Max Structural Discrepancy: {discrepancy:.5e} microns")
