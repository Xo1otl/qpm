# Overview
Cascaded THG prediction via **Coupled-Wave Equations**, first-order **Picard**, and **Wide-Domain Analytic Integration**.

# Coupled-Wave Equations
$$
\frac{d A_1}{dz} = i \left[ \kappa_{SHG}(z) A_2 A_1^* e^{i\Delta k_{SHG} z} + \kappa_{SFG}(z) A_3 A_2^* e^{i\Delta k_{SFG} z} \right] \\
\frac{d A_2}{dz} = i \left[ \kappa_{SHG}(z) A_1^2 e^{-i\Delta k_{SHG} z} + 2 \kappa_{SFG}(z) A_3 A_1^* e^{i\Delta k_{SFG} z} \right] \\
\frac{d A_3}{dz} = i \left[ 3 \kappa_{SFG}(z) A_1 A_2 e^{-i\Delta k_{SFG} z} \right]
$$
$\Delta k$: Phase mismatches. $\kappa(z)$: Spatially-varying coupling.

# Interaction picture
Rotation $\boldsymbol{B}(z) = e^{i\boldsymbol{L}z} \boldsymbol{A}(z)$.

$$\frac{d\boldsymbol{B}}{dz} = i\boldsymbol{L}\boldsymbol{B} + i\boldsymbol{N}(\boldsymbol{B}, z)$$

$$\boldsymbol{L} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & \Delta k_{SHG} & 0 \\ 0 & 0 & \Delta k_{SHG} + \Delta k_{SFG} \end{pmatrix}$$

$$\boldsymbol{N} = \begin{pmatrix}
\kappa_{SHG}(z)B_1^\ast B_2 + \kappa_{SFG}(z)B_2^\ast B_3 \\
\kappa_{SHG}(z)B_1^2 + 2\kappa_{SFG}(z)B_1^\ast B_3 \\
3\kappa_{SFG}(z)B_1 B_2
\end{pmatrix}$$

# Hamiltonian Form (If $\kappa$ is constant)
Hamiltonian $\dot{\boldsymbol{B}} = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} (K_{LIN} + K_{NL})$ with $\boldsymbol{J} = \text{diag}(1, 2, 3)$:
$$K_{NL} = \frac{\kappa_{SHG}(z)}{2} \left( B_1^2 B_2^* + c.c. \right) + \kappa_{SFG}(z) \left( B_1 B_2 B_3^* + c.c. \right)$$
$$K_{LIN} = \frac{\Delta k_{SHG}}{2} |B_2|^2 + \frac{\Delta k_{SHG} + \Delta k_{SFG}}{3} |B_3|^2$$

# Integral Form
$$\boldsymbol{B}(z_n+h) = e^{i\boldsymbol{L}h} \boldsymbol{B}(z_n) + i \int_0^h e^{i\boldsymbol{L}(h-\tau')} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau'), \boldsymbol{B}^*(z_n+\tau'), z_n+\tau') d\tau'$$

# Analytic Integration
Local Undepleted Approximation.
$$\boldsymbol{B}^{(0)}(z_n + \tau') = e^{i\boldsymbol{L}\tau'}\boldsymbol{B}(z_n)$$

Finite Domain Fourier Transform:
$$\mathcal{F}[\kappa](\omega) = \int_{0}^{h} \kappa(z_0 + \tau) e^{i\omega \tau} d\tau$$

Step Update:
$$\boldsymbol{B}(z_0 + h) \approx e^{i\boldsymbol{L}h} \boldsymbol{B}(z_0) + \boldsymbol{B}_{NL}^{step}$$

$$
\boldsymbol{B}_{NL}^{step} = i e^{i\boldsymbol{L}h}
\begin{pmatrix}
B_{1}^* B_{2} \mathcal{F}[\kappa_{SHG}](\Delta k_{SHG}) + B_{2}^* B_{3} \mathcal{F}[\kappa_{SFG}](\Delta k_{SFG}) \\
B_{1}^2 \mathcal{F}[\kappa_{SHG}](-\Delta k_{SHG}) + 2 B_{1}^* B_3 \mathcal{F}[\kappa_{SFG}](\Delta k_{SFG}) \\
3 B_{1} B_2 \mathcal{F}[\kappa_{SFG}](-\Delta k_{SFG}) \\
\end{pmatrix}_{\boldsymbol{B}=\boldsymbol{B}(z_0)}
$$
