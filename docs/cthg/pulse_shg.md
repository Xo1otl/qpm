# Overview
SSFM for SHG solves **Coupled-Wave Equations** via **Symmetric Splitting Algorithm**. Validity limited by **Convergence & Constraints**.

# Coupled-Wave Equations
Interaction between Fundamental ($A_1$) and Second Harmonic ($A_2$).

$$
\frac{\partial A_1}{\partial z} = \underbrace{\left( -\frac{1}{v_{g1}} \frac{\partial}{\partial t} - \frac{i \beta_{2,1}}{2} \frac{\partial^2}{\partial t^2} \right) A_1}_{\hat{D}_1 A_1} + i \kappa^* A_2 A_1^* e^{i \Delta k z}
$$

$$
\frac{\partial A_2}{\partial z} = \underbrace{\left( -\frac{1}{v_{g2}} \frac{\partial}{\partial t} - \frac{i \beta_{2,2}}{2} \frac{\partial^2}{\partial t^2} \right) A_2}_{\hat{D}_2 A_2} + i \kappa A_1^2 e^{-i \Delta k z}
$$

* $\Delta k = k_2 - 2k_1$
* $\kappa$: Nonlinear coefficient

# Symmetric Splitting Algorithm
Scheme: Linear ($h/2$) $\to$ Nonlinear ($h$) $\to$ Linear ($h/2$).

### Step I: Linear Half-Step ($h/2$)
$$
\tilde{A}_j(z_n, \omega) = \mathcal{F}[A_j(z_n, t)]
$$
$$
\tilde{A}_j^{(1)}(\omega) = \tilde{A}_j(z_n, \omega) \exp\left[ i \left( \frac{\omega}{v_{gj}} + \frac{\beta_{2,j}\omega^2}{2} \right) \frac{h}{2} \right]
$$
$$
A_j^{(1)}(t) = \mathcal{F}^{-1}[\tilde{A}_j^{(1)}(\omega)]
$$

### Step II: Nonlinear Step ($h$)
Assumption: Negligible nonlinear loss. Fields constant over $h$ (Euler approximation).

**Effective Coupling Coefficient**
$$
\tilde{\kappa}(\Delta k) \equiv \int_{z_n}^{z_{n+1}} \kappa(z) e^{i \Delta k z} dz
$$

**Update Equations**
$$
A_1^{(2)}(t) = A_1^{(1)}(t) + i A_2^{(1)}(t) (A_1^{(1)}(t))^* \tilde{\kappa}(+\Delta k)
$$
$$
A_2^{(2)}(t) = A_2^{(1)}(t) + i (A_1^{(1)}(t))^2 \tilde{\kappa}(-\Delta k)
$$

### Step III: Linear Half-Step ($h/2$)
$$
\tilde{A}_j^{(2)}(\omega) = \mathcal{F}[A_j^{(2)}(t)]
$$
$$
\tilde{A}_j(z_{n+1}, \omega) = \tilde{A}_j^{(2)}(\omega) \exp\left[ i \left( \frac{\omega}{v_{gj}} + \frac{\beta_{2,j}\omega^2}{2} \right) \frac{h}{2} \right]
$$
$$
A_j(z_{n+1}, t) = \mathcal{F}^{-1}[\tilde{A}_j(z_{n+1}, \omega)]
$$

# Convergence & Constraints

**Walk-off**
$$
h \ll L_{walk} = \frac{\tau_p}{|v_{g1}^{-1} - v_{g2}^{-1}|}
$$

**Dispersion**
$$
h \ll L_{disp} = \frac{\tau_p^2}{|\beta_2|}
$$

# Allowed
**QPM Period**
$$
h > \Lambda
$$
