import numpy as np

# Physical constants
epsilon0_F_um = 8.854e-18  # [F/um]
c_um_s = 2.9979e14  # [um/s]
Z0 = 376.73  # [Ohm] (derived as 1/(epsilon0 * c), scale independent)

# Inputs in um scale
FWHM_um = 30.0  # [um]
d33_um_V = 1.38e-5  # [um/V] (corresponds to 13.8 pm/V)
n = 2.2  # [dimensionless]
lambda_fund_um = 1.55  # [um]

# Effective area calculation in um scale
w_um = FWHM_um / np.sqrt(2 * np.log(2))  # [um]
A_eff_um2 = np.pi * w_um**2  # [um^2]

# Kappa calculation using um scale variables
# Units: ([um/V] / ([um] * [um])) * [V/W^1/2] = [W^-1/2 um^-1]
numerator = np.sqrt(2 * np.pi) * d33_um_V
denominator = lambda_fund_um * np.sqrt(A_eff_um2) * (n**1.5)
kappa_per_um = (numerator / denominator) * np.sqrt(Z0)  # [W^-1/2 um^-1]

# Output
print(f"A_eff: {A_eff_um2:.2f} um^2")
print(f"Calculated Kappa: {kappa_per_um * 1e4:.4f} W^-1/2 cm^-1")  # Convert [um^-1] to [cm^-1]
