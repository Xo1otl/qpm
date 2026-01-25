# Overview
Cascaded THG prediction via **Coupled-Wave Equations**, first-order **Picard**, **Aperiodic Sign Flipping**, and **Poly-Domain Analytic Integration**.

# Coupled-Wave Equations
$$
\frac{d A_1}{dz} = i \left[ \kappa_{SHG} A_2 A_1^* e^{i\Delta k_{SHG} z} + \kappa_{SFG} A_3 A_2^* e^{i\Delta k_{SFG} z} \right] \\
\frac{d A_2}{dz} = i \left[ \kappa_{SHG} A_1^2 e^{-i\Delta k_{SHG} z} + 2 \kappa_{SFG} A_3 A_1^* e^{i\Delta k_{SFG} z} \right] \\
\frac{d A_3}{dz} = i \left[ 3 \kappa_{SFG} A_1 A_2 e^{-i\Delta k_{SFG} z} \right]
$$
$\Delta k$: Phase mismatches. $\kappa(z)$: Coupling coefficients.

# Hamiltonian Form
Rotation $\boldsymbol{B}(z) = e^{i\boldsymbol{L}z} \boldsymbol{A}(z)$.
$$\boldsymbol{L} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & \Delta k_{SHG} & 0 \\ 0 & 0 & \Delta k_{SHG} + \Delta k_{SFG} \end{pmatrix}$$

Hamiltonian $\dot{\boldsymbol{B}} = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} (K_{LIN} + K_{NL})$ with $\boldsymbol{J} = \text{diag}(1, 2, 3)$:
$$K_{NL} = \frac{\kappa_{SHG}}{2} \left( B_1^2 B_2^* + c.c. \right) + \kappa_{SFG} \left( B_1 B_2 B_3^* + c.c. \right)$$
$$K_{LIN} = \frac{\Delta k_{SHG}}{2} |B_2|^2 + \frac{\Delta k_{SHG} + \Delta k_{SFG}}{3} |B_3|^2$$

# Integral Form
$$\boldsymbol{B}(z_n+h) = e^{i\boldsymbol{L}h} \boldsymbol{B}(z_n) + i \int_0^h e^{i\boldsymbol{L}(h-\tau')} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau'), \boldsymbol{B}^*(z_n+\tau')) d\tau'$$

$$N=\left(\begin{matrix}\kappa_{SHG}B_1^\ast B_2+\kappa_{SFG}B_2^\ast B_3\\\kappa_{SHG}B_1^2+2\kappa_{SFG}B_1^\ast B_3\\3\kappa_{SFG}B_1B_2\\\end{matrix}\right)$$

# Picard / Interaction Picture
Linear flow only:
$$\boldsymbol{B}^{(0)}(z_n + \tau') = e^{i\boldsymbol{L}\tau'}\boldsymbol{B}(z_n)$$

Prediction $\boldsymbol{B}_{pred} = e^{i\boldsymbol{L}h_n} \boldsymbol{B}_n + \boldsymbol{B}_{NL}$ with $\phi(\omega, h) = (e^{i\omega h} - 1)/(i\omega)$:

$$
\boldsymbol{B}_{NL} = i e^{i\boldsymbol{L}h_n}
\begin{pmatrix}
\kappa_{SHG} B_{1n}^* B_{2n} \phi(\Delta k_{SHG}, h_n) + \kappa_{SFG} B_{2n}^* B_{3n} \phi(\Delta k_{SFG}, h_n) \\
\kappa_{SHG} B_{1n}^2 \phi(-\Delta k_{SHG}, h_n) + 2 \kappa_{SFG} B_{1n}^* B_{3n} \phi(\Delta k_{SFG}, h_n) \\
3 \kappa_{SFG} B_{1n} B_{2n} \phi(-\Delta k_{SFG}, h_n)
\end{pmatrix}
$$

# Aperiodic Sign Flipping
Lengths $\{h_n\}$, signs $s_n$ (e.g., $s_n = (-1)^n$).
$$\kappa_n = s_n |\kappa|$$

Step Operator:
$$\hat{\mathcal{U}}(h, s)[\boldsymbol{B}] \equiv e^{i\boldsymbol{L}h} \boldsymbol{B} + \boldsymbol{B}_{NL}(\boldsymbol{B}, h, s|\kappa|)$$

System Evolution:
$$\boldsymbol{B}_{final} = \left( \bigcirc_{n=0}^{N-1} \hat{\mathcal{U}}(h_n, s_n) \right) \boldsymbol{B}_{0}$$

# Poly-Domain Analytic Integration
Merge $M$ domains. Block length $H = \sum_{j=0}^{M-1} h_j$.
Assumption: Slow envelope $\boldsymbol{B}$, invariant $|\kappa|$.

Structure Factor $\Psi(\omega)$ (Geometric scalar sum):
$$\Psi(\omega) = \sum_{j=0}^{M-1} s_j e^{i\omega Z_j} \phi(\omega, h_j)$$
$Z_0 = 0$, $Z_j = \sum_{k=0}^{j-1} h_k$. $\omega \in \{\pm \Delta k_{SHG}, \pm \Delta k_{SFG}\}$.

Block Update:
$$\boldsymbol{B}(z_0 + H) \approx e^{i\boldsymbol{L}H} \boldsymbol{B}(z_0) + \boldsymbol{B}_{NL}^{block}$$

$$
\boldsymbol{B}_{NL}^{block} = i e^{i\boldsymbol{L}H}
\begin{pmatrix}
|\kappa_{SHG}| B_{1}^* B_{2} \Psi(\Delta k_{SHG}) + |\kappa_{SFG}| B_{2}^* B_{3} \Psi(\Delta k_{SFG}) \\
|\kappa_{SHG}| B_{1}^2 \Psi(-\Delta k_{SHG}) + 2 |\kappa_{SFG}| B_{1}^* B_{3} \Psi(\Delta k_{SFG}) \\
3 |\kappa_{SFG}| B_{1} B_{2} \Psi(-\Delta k_{SFG})
\end{pmatrix}_{\boldsymbol{B}=\boldsymbol{B}_0}
$$
