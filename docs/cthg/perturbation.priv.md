# Overview
Model for **Cascaded THG (SHG+SFG)** via coupled-wave equations. Transformed to **Canonical Form** to remove $z$-dependence. Solved using **Integral Form** and **Picard Iteration** (1st-order perturbation).

# Coupled-Wave Equations
$$\frac{d A_1}{dz} = i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right] \\ \frac{d A_2}{dz} = i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \\ \frac{d A_3}{dz} = i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right]$$
$\boldsymbol{A}$: amplitude ($\sum |A_j|^2 = const$). $\kappa$: coupling. $\Delta k$: mismatch.

# Canonical Form
Rotation to remove fast phase:
$$\boldsymbol{B}(z) = e^{i\boldsymbol{L}z} \boldsymbol{A}(z) \quad \text{where} \quad \boldsymbol{L} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & \Delta k_1 & 0 \\ 0 & 0 & \Delta k_1 + \Delta k_2 \end{pmatrix}$$

Hamiltonian dynamics $\dot{\boldsymbol{B}} = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} (K_{LIN} + K_{NL})$:
$$K = \underbrace{\frac{\kappa}{2} \left( B_1^2 B_2^* + c.c. \right) + \kappa \left( B_1 B_2 B_3^* + c.c. \right)}_{K_{NL}} + \underbrace{\frac{\Delta k_1}{2} |B_2|^2 + \frac{\Delta k_1 + \Delta k_2}{3} |B_3|^2}_{K_{LIN}}$$
QPM: $\kappa(z)$ sign inverted with period $\Lambda \approx 2\pi/\Delta k$.

# Integral Form
Separation of linear/nonlinear terms:
$$\frac{d\boldsymbol{B}}{dz} = i \left( \boldsymbol{L}\boldsymbol{B} + \boldsymbol{N}(\boldsymbol{B}, \boldsymbol{B}^*) \right)$$
$$i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{LIN} = i \boldsymbol{L} \boldsymbol{B}, \quad i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{NL} = i \boldsymbol{N}$$

Exact solution:
$$\boldsymbol{B}(z_n+h) = e^{i\boldsymbol{L}h} \boldsymbol{B}(z_n) + i \int_0^h e^{i\boldsymbol{L}(h-\tau')} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau'), \boldsymbol{B}^*(z_n+\tau')) d\tau'$$

# Picard Iteration (Perturbative)
Method: **Picard iteration (Successive Approximation)**.
Approximate nonlinear term with linear evolution $\boldsymbol{B}^{(0)}(z_n + \tau') = e^{i\boldsymbol{L}\tau'}\boldsymbol{B}(z_n)$.
(**First-order perturbation**: assumes slow amplitude change vs phase).

1.  **Integral Function**: $\phi(\omega, h) = (e^{i\omega h} - 1)/(i\omega)$ if $\omega \neq 0$, else $h$.

2.  **Nonlinear Evolution**:
$$B_{NL, 1} = i\kappa_n e^{il_1 h_n} \left[ B_{1n}^* B_{2n} \phi(l_2-2l_1, h_n) + B_{2n}^* B_{3n} \phi(l_3-l_2-l_1, h_n) \right] \\ B_{NL, 2} = i\kappa_n e^{il_2 h_n} \left[ B_{1n}^2 \phi(2l_1-l_2, h_n) + 2 B_{1n}^* B_{3n} \phi(l_3-l_1-l_2, h_n) \right] \\ B_{NL, 3} = i \, 3\kappa_n e^{il_3 h_n} \left[ B_{1n} B_{2n} \phi(l_1+l_2-l_3, h_n) \right]$$

3.  **Prediction**:
$$\boldsymbol{B}_{pred} = e^{i\boldsymbol{l}h_n} \boldsymbol{B}_n + \boldsymbol{B}_{NL}$$
