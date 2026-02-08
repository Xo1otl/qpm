# Overview
Derivation of **Prediction Scheme** via **Local Fourier Coupling Coefficient** from **Coupled-Wave Equations**.

# Coupled-Wave Equations
$$
\frac{d A_1}{dz} = i \left[ \kappa_{SHG}(z) A_2 A_1^* e^{i\Delta k_{SHG} z} + \kappa_{SFG}(z) A_3 A_2^* e^{i\Delta k_{SFG} z} \right] \\
\frac{d A_2}{dz} = i \left[ \kappa_{SHG}(z) A_1^2 e^{-i\Delta k_{SHG} z} + 2 \kappa_{SFG}(z) A_3 A_1^* e^{i\Delta k_{SFG} z} \right] \\
\frac{d A_3}{dz} = i \left[ 3 \kappa_{SFG}(z) A_1 A_2 e^{-i\Delta k_{SFG} z} \right]
$$

# Prediction Scheme
Fourier Coefficient.
$$\mathcal{F}[\kappa](\omega) = \int_{z_n}^{z_n+h} \kappa(z) e^{i\omega z} dz$$

Fixed amplitude within integral $z \in [z_n, z_{n+1}]$.
$$A_j(z) \approx A_j(z_n)$$

Substitute approximation into integrated ODEs.

$$
A_1(z_{n+1}) = A_1 + i \left[ A_2 A_1^* \mathcal{F}[\kappa_{SHG}](\Delta k_{SHG}) + A_3 A_2^* \mathcal{F}[\kappa_{SFG}](\Delta k_{SFG}) \right] \\
A_2(z_{n+1}) = A_2 + i \left[ A_1^2 \mathcal{F}[\kappa_{SHG}](-\Delta k_{SHG}) + 2 A_3 A_1^* \mathcal{F}[\kappa_{SFG}](\Delta k_{SFG}) \right] \\
A_3(z_{n+1}) = A_3 + i \left[ 3 A_1 A_2 \mathcal{F}[\kappa_{SFG}](-\Delta k_{SFG}) \right] \\
$$
