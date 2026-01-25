# Overview
Cascaded THG prediction via **Coupled-Wave Equations** and first-order **Picard** with **Aperiodic Sign Flipping**.

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

# Aperiodic Propagation
Lengths $\{h_n\}$, signs $s_n$ (e.g., $s_n = (-1)^n$).
$$\kappa_n = s_n |\kappa|$$

Step Operator:
$$\hat{\mathcal{U}}(h, s)[\boldsymbol{B}] \equiv e^{i\boldsymbol{L}h} \boldsymbol{B} + \boldsymbol{B}_{NL}(\boldsymbol{B}, h, s|\kappa|)$$

System Evolution:
$$\boldsymbol{B}_{final} = \left( \bigcirc_{n=0}^{N-1} \hat{\mathcal{U}}(h_n, s_n) \right) \boldsymbol{B}_{0}$$
$$\bigcirc_{n=0}^{N-1} \hat{\mathcal{U}}_n \equiv \hat{\mathcal{U}}_{N-1} \circ \dots \circ \hat{\mathcal{U}}_0$$

# Task
Propose a theoretical optimization that specifically exploit the invariance of $|\kappa|$

# Answer
Theoretical Optimization: Poly-Domain Analytic Integration (PDAI)
Proposal:
Merge $M$ consecutive domains (segments with varying $h_n$ and $s_n$) into a single "Super-Step" update. Since the envelope $\boldsymbol{B}$ varies slowly compared to the phase $\Delta k$ and sign flips $s_n$, you can extend the validity of the first-order Picard approximation over a block of domains.
By exploiting the invariance of $|\kappa|$ and the linearity of the integral operator with respect to the sign sequence $s_n$, you can factorize the nonlinear update into a purely geometric scalar sum (Structure Factor) and a single vector update.
1. The Super-Step Operator
Define a block length $H = \sum_{j=0}^{M-1} h_j$. Within the first-order Picard approximation, the field evolution over this block is:
$$\boldsymbol{B}(z_0 + H) \approx e^{i\boldsymbol{L}H} \boldsymbol{B}(z_0) + \boldsymbol{B}_{NL}^{block}$$
The nonlinear correction term integrates over the piecewise-constant coupling profile:
$$\boldsymbol{B}_{NL}^{block} = i e^{i\boldsymbol{L}H} \int_0^H e^{-i\boldsymbol{L}\tau} \boldsymbol{N}(\boldsymbol{B}^{(0)}(\tau), \boldsymbol{B}^{(0)*}(\tau)) d\tau$$
2. Factorization of Invariant Coupling
Substitute the explicit form of $\boldsymbol{N}$. Since $\kappa(\tau) = s(\tau)|\kappa|$, the magnitude $|\kappa|$ factors out. The integral reduces to a sum of integrals over each subdomain $j$, where the only changing variables are the sign $s_j$ and the cumulative phase.
Define the Effective Structure Factor $\Psi(\omega)$ for the block:
$$\Psi(\omega) = \sum_{j=0}^{M-1} s_j e^{i\omega Z_j} \phi(\omega, h_j)$$
Where:
$Z_0 = 0$, $Z_j = \sum_{k=0}^{j-1} h_k$ (Cumulative depth within block)
$\omega \in \{\pm \Delta k_{SHG}, \pm \Delta k_{SFG}\}$
3. Optimized Update Equations
The update for the entire block requires only one evaluation of the nonlinear vector, scaled by the pre-computed Structure Factors.
$$\boldsymbol{B}_{NL}^{block} = i e^{i\boldsymbol{L}H}
\begin{pmatrix}
|\kappa_{SHG}| B_{1}^* B_{2} \Psi(\Delta k_{SHG}) + |\kappa_{SFG}| B_{2}^* B_{3} \Psi(\Delta k_{SFG}) \\
|\kappa_{SHG}| B_{1}^2 \Psi(-\Delta k_{SHG}) + 2 |\kappa_{SFG}| B_{1}^* B_{3} \Psi(\Delta k_{SFG}) \\
3 |\kappa_{SFG}| B_{1} B_{2} \Psi(-\Delta k_{SFG})
\end{pmatrix}_{\boldsymbol{B}=\boldsymbol{B}_0}$$
4. Computational Advantage
Reduction of Vector Ops: Instead of calculating the vector $\boldsymbol{B}_{NL}$ (which involves expensive complex multiplications) $M$ times, you calculate it once per block.
Pre-computation: If the aperiodic sequence $\{h_n, s_n\}$ is known a priori, $\Psi(\omega)$ can be pre-calculated for the entire propagation length, completely decoupling the domain geometry from the field propagation loop.
Error Bound: The approximation holds as long as the conversion efficiency per block remains in the perturbative regime ($|\boldsymbol{B}_{NL}^{block}| \ll |\boldsymbol{B}|$).
Next Step
Would you like me to derive the error term for this Poly-Domain approach to determine the maximum optimal block size $M$?
