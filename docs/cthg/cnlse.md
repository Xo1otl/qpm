# Overview
Derivation of **Super-Step SSFM(S4M)** for **Coupled-Wave Equations** accelerated by LECC.

# NLSE Coupled-Wave Equations
System defined by operator splitting of **Linear** ($\hat{D}$) and **Nonlinear** ($\hat{N}$) effects.
$$\frac{\partial \mathbf{A}}{\partial z} = (\hat{D} + \hat{N}) \mathbf{A}$$
Where $\mathbf{A} = [A_1, A_2]^T$. Solution over step $h$ via symmetric splitting:
$$\mathbf{A}(z+h, t) \approx e^{\frac{h}{2}\hat{D}} e^{h\hat{N}} e^{\frac{h}{2}\hat{D}} \mathbf{A}(z, t)$$

# Linear Operator (Dispersion/Walk-off)
Diagonal in frequency domain.
$$\hat{D}_j = - \frac{1}{v_{gj}} \frac{\partial}{\partial t} - \frac{i \beta_{2,j}}{2} \frac{\partial^2}{\partial t^2}$$

Solved via FFT:
$$\tilde{A}_j(z, \omega) = \text{FFT}[A_j(z, t)]$$
$$\tilde{A}_j^{(1)}(z, \omega) = \tilde{A}_j(z, \omega) \exp\left[ -i \left( \frac{\omega}{v_{gj}} + \frac{\beta_{2,j}\omega^2}{2} \right) \frac{h}{2} \right]$$
$$A_j^{(1)}(z, t) = \text{IFFT}[\tilde{A}_j^{(1)}(z, \omega)]$$

# Local Effective Coupling Coefficient
Integration of $\hat{N}$ strictly in spatial domain. Time $t$ parameterized; envelope $A_j$ frozen relative to $\kappa(z)$.

**LECC Definition:**
$$\mathcal{F}[\kappa](\Omega) \equiv \int_{z}^{z+h} \kappa(z') e^{i \Omega z'} dz'$$

**Update Equations (SHG):**
Substitute frozen amplitudes $A_j^{(1)}$ into integrated ODEs.
$$A_1^{(2)}(z, t) = A_1^{(1)} + i A_2^{(1)} (A_1^{(1)})^* \mathcal{F}[\kappa](-\Delta k)$$
$$A_2^{(2)}(z, t) = A_2^{(1)} + i (A_1^{(1)})^2 \mathcal{F}[\kappa](+\Delta k)$$

# Convergence Criterion
Step size $h$ constrained by **Walk-off** and **Dispersion**, not Grating Period $\Lambda$.
$$h \ll \min \left( \frac{\tau_p}{|v_{g1}^{-1} - v_{g2}^{-1}|}, \frac{\tau_p^2}{|\beta_2|} \right)$$
