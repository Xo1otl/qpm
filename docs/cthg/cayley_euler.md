# Overview
Comparison: **Explicit Local Fourier** (non-conservative baseline) and **Cayley-Euler Scheme** (energy-preserving proposal).

# Explicit Local Fourier
Additive, $O(N)$, non-symplectic. Energy drifts.

**Update Rule**
$$\mathbf{A}_{n+1} = \mathbf{A}_n + i \Delta_n$$

**Component Updates**
$$
\begin{aligned}
A_1^{(n+1)} &= A_1 + i \left[ A_2 A_1^* \kappa_{\text{leff}, SHG} + A_3 A_2^* \kappa_{\text{leff}, SFG} \right] \\
A_2^{(n+1)} &= A_2 + i \left[ A_1^2 \kappa_{\text{leff}, SHG}^* + 2 A_3 A_1^* \kappa_{\text{leff}, SFG} \right] \\
A_3^{(n+1)} &= A_3 + i \left[ 3 A_1 A_2 \kappa_{\text{leff}, SFG}^* \right]
\end{aligned}
$$
*(RHS frozen at step $n$)*

# Cayley-Euler Scheme
Multiplicative, geometric integrator via Cayley Transform. Unconditionally stable, conserves invariant $\sum |u_k|^2$. 1st-order accuracy.

**Coordinate Transformation**
Identity basis (1:1 ODE system match).
$$u_1 = A_1, \quad u_2 = A_2, \quad u_3 = A_3$$

**Update Rule**
$$\mathbf{u}_{n+1} = \left( \mathbf{I} - \frac{1}{2}\mathbf{\Omega}_n \right)^{-1} \left( \mathbf{I} + \frac{1}{2}\mathbf{\Omega}_n \right) \mathbf{u}_n$$

**Interaction Matrix $\mathbf{\Omega}_n$**
Skew-Hermitian ($\mathbf{\Omega}^\dagger = -\mathbf{\Omega}$). Frozen at $z_n$.
$$
\mathbf{\Omega}_n = i \begin{pmatrix} 
0 & \mu_{12} & \mu_{13} \\
\mu_{12}^* & 0 & \mu_{23} \\
\mu_{13}^* & \mu_{23}^* & 0
\end{pmatrix}
$$

**Effective Couplings (Frozen)**
Using $\kappa_{\text{leff}}(\Delta k) = \int_z^{z+h} \kappa e^{i\Delta k \xi} d\xi$:

* **SHG ($u_1 \leftrightarrow u_2$):**
    $$\mu_{12} = \kappa_{\text{leff}, SHG}(\Delta k_{SHG}) \cdot A_1^*(z_n)$$

* **SFG ($u_1 \leftrightarrow u_3$):**
    $$\mu_{13} = \kappa_{\text{leff}, SFG}(\Delta k_{SFG}) \cdot A_2^*(z_n)$$

* **SFG ($u_2 \leftrightarrow u_3$):**
    $$\mu_{23} = 2 \cdot \kappa_{\text{leff}, SFG}(\Delta k_{SFG}) \cdot A_1^*(z_n)$$
