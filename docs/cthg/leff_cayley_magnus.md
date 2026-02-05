# Overview
Derivation of **Coupled-Wave Equations**, **Prediction Scheme** via **Local Fourier Coupling Coefficient**, and **Cayley-Magnus Scheme**.

# Coupled-Wave Equations
$$
\frac{d A_1}{dz} = i \left[ \kappa_{SHG}(z) A_2 A_1^* e^{i\Delta k_{SHG} z} + \kappa_{SFG}(z) A_3 A_2^* e^{i\Delta k_{SFG} z} \right] \\
\frac{d A_2}{dz} = i \left[ \kappa_{SHG}(z) A_1^2 e^{-i\Delta k_{SHG} z} + 2 \kappa_{SFG}(z) A_3 A_1^* e^{i\Delta k_{SFG} z} \right] \\
\frac{d A_3}{dz} = i \left[ 3 \kappa_{SFG}(z) A_1 A_2 e^{-i\Delta k_{SFG} z} \right]
$$

# Prediction Scheme
Fourier Coefficient.
$$\mathcal{F}[\kappa](\omega) = \int_{z_n}^{z_n+h} \kappa(z) e^{i\omega z} dz$$

Assumption: Fixed amplitude within integral $z \in [z_n, z_{n+1}]$.
$$A_j(z) \approx A_j(z_n)$$

Substitute approximation into integrated ODEs.

$$
A_1(z_{n+1}) = A_1 + i \left[ A_2 A_1^* \mathcal{F}[\kappa_{SHG}](\Delta k_{SHG}) + A_3 A_2^* \mathcal{F}[\kappa_{SFG}](\Delta k_{SFG}) \right] \\
A_2(z_{n+1}) = A_2 + i \left[ A_1^2 \mathcal{F}[\kappa_{SHG}](-\Delta k_{SHG}) + 2 A_3 A_1^* \mathcal{F}[\kappa_{SFG}](\Delta k_{SFG}) \right] \\
A_3(z_{n+1}) = A_3 + i \left[ 3 A_1 A_2 \mathcal{F}[\kappa_{SFG}](-\Delta k_{SFG}) \right] \\
$$

# Cayley-Magnus Scheme
Multiplicative, geometric, energy-preserving, 2nd-order.

## Predictor Step (Midpoint)
Estimate $\mathbf{A}_{n+1/2}$ via Explicit Local Fourier (step $h/2$).

$$
\mathbf{A}_{n+1/2} = \mathbf{A}_n + \frac{i}{2} \Delta_n
$$

$\Delta_n$ explicit update at $\mathbf{A}_n$:
$$
\begin{aligned}
\Delta_{n,1} &= A_2 A_1^* \kappa_{\text{leff}, SHG} + A_3 A_2^* \kappa_{\text{leff}, SFG} \\
\Delta_{n,2} &= A_1^2 \kappa_{\text{leff}, SHG}^* + 2 A_3 A_1^* \kappa_{\text{leff}, SFG} \\
\Delta_{n,3} &= 3 A_1 A_2 \kappa_{\text{leff}, SFG}^*
\end{aligned}
$$

## Magnus Generator
Skew-Hermitian matrix $\mathbf{\Omega}_{n+1/2}$ at $z_{n+1/2}$.

$$
\mathbf{\Omega}_{n+1/2} = i \begin{pmatrix} 
0 & \mu_{12} & \mu_{13} \\
\mu_{12}^* & 0 & \mu_{23} \\
\mu_{13}^* & \mu_{23}^* & 0
\end{pmatrix}
$$

**Midpoint Effective Couplings** using $\mathbf{A}_{n+1/2}$:
* $\mu_{12} = \kappa_{\text{leff}, SHG} \cdot A_1^*(z_{n+1/2})$
* $\mu_{13} = \kappa_{\text{leff}, SFG} \cdot A_2^*(z_{n+1/2})$
* $\mu_{23} = 2 \cdot \kappa_{\text{leff}, SFG} \cdot A_1^*(z_{n+1/2})$

## Update Rule (Cayley)
Propagate $z_n \to z_{n+1}$.

$$
\mathbf{A}_{n+1} = \left( \mathbf{I} - \frac{1}{2}\mathbf{\Omega}_{n+1/2} \right)^{-1} \left( \mathbf{I} + \frac{1}{2}\mathbf{\Omega}_{n+1/2} \right) \mathbf{A}_n
$$
