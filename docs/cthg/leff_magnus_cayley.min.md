# Overview
Derivation of **Coupled-Wave Equations**, **Local Fourier Coefficients**, **Prediction Scheme**, and **Magnus-Cayley Scheme**.

# Coupled-Wave Equations
$$
\frac{d A_1}{dz} = i \left[ \kappa_{SHG}(z) A_2 A_1^* e^{i\Delta k_{SHG} z} + \kappa_{SFG}(z) A_3 A_2^* e^{i\Delta k_{SFG} z} \right] \\
\frac{d A_2}{dz} = i \left[ \kappa_{SHG}(z) A_1^2 e^{-i\Delta k_{SHG} z} + 2 \kappa_{SFG}(z) A_3 A_1^* e^{i\Delta k_{SFG} z} \right] \\
\frac{d A_3}{dz} = i \left[ 3 \kappa_{SFG}(z) A_1 A_2 e^{-i\Delta k_{SFG} z} \right]
$$

# Local Fourier Coefficients
Integral of oscillating terms over step $h$:
$$\mathcal{F}[\kappa](\omega, h) = \int_{z_n}^{z_n+h} \kappa(z) e^{i\omega z} dz$$

# Prediction Scheme
Estimate $\mathbf{A}_{n+1/2}$ via Explicit Local Fourier (step $h/2$).
$$\mathbf{A}_{n+1/2} = \mathbf{A}_n + i \mathbf{\Delta}_n$$

Update components using half-step coefficients $\mathcal{F}(h/2)$:
$$
\begin{aligned}
\Delta_{n,1} &= A_2 A_1^* \mathcal{F}[\kappa_{SHG}](\Delta k_{SHG}, h/2) + A_3 A_2^* \mathcal{F}[\kappa_{SFG}](\Delta k_{SFG}, h/2) \\
\Delta_{n,2} &= A_1^2 \mathcal{F}[\kappa_{SHG}](-\Delta k_{SHG}, h/2) + 2 A_3 A_1^* \mathcal{F}[\kappa_{SFG}](\Delta k_{SFG}, h/2) \\
\Delta_{n,3} &= 3 A_1 A_2 \mathcal{F}[\kappa_{SFG}](-\Delta k_{SFG}, h/2)
\end{aligned}
$$

# Magnus-Cayley Scheme
## Magnus Generator
First-order expansion $\mathbf{\Omega}_1$ via Midpoint Rule:
$$\mathbf{\Omega}_1 = \int_{z_n}^{z_{n+1}} \mathbf{M}(\tau, \mathbf{A}_{n+1/2}) d\tau$$

Skew-Hermitian matrix $\mathbf{\Omega}_{n+1/2}$ constructed using full-step coefficients $\mathcal{F}(h)$:
$$
\mathbf{\Omega}_{n+1/2} = i \begin{pmatrix}
0 & \mu_{12} & \mu_{13} \\
\mu_{12}^* & 0 & \mu_{23} \\
\mu_{13}^* & \mu_{23}^* & 0
\end{pmatrix}
$$

**Effective Couplings** frozen at $\mathbf{A}_{n+1/2}$:
* $\mu_{12} = \mathcal{F}[\kappa_{SHG}](\Delta k_{SHG}, h) \cdot A_1^*(z_{n+1/2})$
* $\mu_{13} = \mathcal{F}[\kappa_{SFG}](\Delta k_{SFG}, h) \cdot A_2^*(z_{n+1/2})$
* $\mu_{23} = 2 \cdot \mathcal{F}[\kappa_{SFG}](\Delta k_{SFG}, h) \cdot A_1^*(z_{n+1/2})$

## Update Rule
Cayley transform mapping $\mathfrak{u}(3) \to U(3)$ (Unitary, Energy-Preserving):
$$
\mathbf{A}_{n+1} = \left( \mathbf{I} - \frac{1}{2}\mathbf{\Omega}_{n+1/2} \right)^{-1} \left( \mathbf{I} + \frac{1}{2}\mathbf{\Omega}_{n+1/2} \right) \mathbf{A}_n
$$
