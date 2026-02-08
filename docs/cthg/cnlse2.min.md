# Overview
**Coupled-Wave Equations (CNLSE)**. **Linear Operator**. **Effective Coupling Coefficient**. **Convergence**.

# Coupled-Wave Equations (CNLSE)
Operator splitting ($\hat{D} + \hat{N}(z)$). Use symmetric splitting.

$$
\frac{\partial \mathbf{A}}{\partial z} = (\hat{D} + \hat{N}(z)) \mathbf{A}
$$

**Symmetric Splitting**:
$$\mathbf{A}(z_{n+1}, t) \approx e^{\frac{h}{2}\hat{D}} \hat{\mathcal{U}}_N(z_n, h) e^{\frac{h}{2}\hat{D}} \mathbf{A}(z_n, t)$$

# Linear Operator
**Time Domain**:
$$
\hat{D}_j = - \frac{1}{v_{gj}} \frac{\partial}{\partial t} - \frac{i \beta_{2,j}}{2} \frac{\partial^2}{\partial t^2}
$$

**Spectral Propagator**:
$$
\tilde{A}_j^{(1)}(z_n, \omega) = \mathcal{F}[A_j(z_n, t)] \exp\left[ +i \left( \frac{\omega}{v_{gj}} + \frac{\beta_{2,j}\omega^2}{2} \right) \frac{h}{2} \right]
$$
$$
A_j^{(1)}(z_n, t) = \mathcal{F}^{-1}[\tilde{A}_j^{(1)}(z_n, \omega)]
$$

# Effective Coupling Coefficient
**Definition**:
$$
\tilde{\kappa}(\Delta k) \equiv \int_{z_n}^{z_{n+1}} \kappa(z) e^{i \Delta k z} dz
$$

**Update Equations (SHG)**:
$$
A_1^{(2)} = A_1^{(1)} + i A_2^{(1)} (A_1^{(1)})^* \tilde{\kappa}(-\Delta k)
$$
$$
A_2^{(2)} = A_2^{(1)} + i (A_1^{(1)})^2 \tilde{\kappa}(+\Delta k)
$$

# Convergence Criterion
$h$ constrained by pulse dynamics.

**Constraints**:
$$
h \ll \min \left( \frac{\tau_p}{|v_{g1}^{-1} - v_{g2}^{-1}|}, \frac{\tau_p^2}{|\beta_2|} \right)
$$

**Allowed**:
$$
h > \Lambda
$$
