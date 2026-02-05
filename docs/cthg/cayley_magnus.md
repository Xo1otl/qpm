# Cayley-Magnus Scheme
Multiplicative, geometric integrator using 2nd-order Magnus Expansion with Cayley Transform. Energy-preserving and 2nd-order accurate.

## 1. Predictor Step (Midpoint Estimation)
Estimate field amplitudes at $z_{n+1/2}$ using the Explicit Local Fourier method with step size $h/2$.

$$
\mathbf{A}_{n+1/2} = \mathbf{A}_n + \frac{i}{2} \Delta_n
$$

Where $\Delta_n$ is the explicit update vector evaluated at $\mathbf{A}_n$:
$$
\begin{aligned}
\Delta_{n,1} &= A_2 A_1^* \kappa_{\text{leff}, SHG} + A_3 A_2^* \kappa_{\text{leff}, SFG} \\
\Delta_{n,2} &= A_1^2 \kappa_{\text{leff}, SHG}^* + 2 A_3 A_1^* \kappa_{\text{leff}, SFG} \\
\Delta_{n,3} &= 3 A_1 A_2 \kappa_{\text{leff}, SFG}^*
\end{aligned}
$$

## 2. Magnus Generator Construction
Construct the skew-Hermitian interaction matrix $\mathbf{\Omega}_{n+1/2}$ evaluated at the midpoint $z_{n+1/2}$.

$$
\mathbf{\Omega}_{n+1/2} = i \begin{pmatrix} 
0 & \mu_{12} & \mu_{13} \\
\mu_{12}^* & 0 & \mu_{23} \\
\mu_{13}^* & \mu_{23}^* & 0
\end{pmatrix}
$$

**Midpoint Effective Couplings**
Using $\kappa_{\text{leff}}$ for full step $h$, but fields from $\mathbf{A}_{n+1/2}$:

* **SHG:** $\mu_{12} = \kappa_{\text{leff}, SHG} \cdot A_1^*(z_{n+1/2})$
* **SFG 1:** $\mu_{13} = \kappa_{\text{leff}, SFG} \cdot A_2^*(z_{n+1/2})$
* **SFG 2:** $\mu_{23} = 2 \cdot \kappa_{\text{leff}, SFG} \cdot A_1^*(z_{n+1/2})$

## 3. Update Rule (Cayley Transform)
Propagate from $z_n$ to $z_{n+1}$ using the midpoint generator.

$$
\mathbf{A}_{n+1} = \left( \mathbf{I} - \frac{1}{2}\mathbf{\Omega}_{n+1/2} \right)^{-1} \left( \mathbf{I} + \frac{1}{2}\mathbf{\Omega}_{n+1/2} \right) \mathbf{A}_n
$$
