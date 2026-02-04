# Prediction Scheme Formulation

## 1. Baseline: Explicit Local Fourier (Non-Conservative)
The original scheme uses an **additive** update. While efficient ($O(N)$), it is not symplectic. The energy (norm) drifts over long distances or with large step sizes $h$, as the update vector is tangential to the energy surface.

**Update Rule:**
$$\mathbf{A}_{n+1} = \mathbf{A}_n + i \Delta_n$$

**Component Updates:**
$$
\begin{aligned}
A_1^{(n+1)} &= A_1 + i \left[ A_2 A_1^* \beta_{SHG} + A_3 A_2^* \beta_{SFG} \right] \\
A_2^{(n+1)} &= A_2 + i \left[ A_1^2 \beta_{SHG}^* + 2 A_3 A_1^* \beta_{SFG} \right] \\
A_3^{(n+1)} &= A_3 + i \left[ 3 A_1 A_2 \beta_{SFG}^* \right]
\end{aligned}
$$
*(All $A$ on RHS are frozen at step $n$)*

---

## 2. Proposed: Quasi-Magnus Cayley Scheme (Energy-Preserving)
This scheme uses a **multiplicative** update via the **Cayley Transform**. It maps the skew-Hermitian interaction matrix to a unitary operator, guaranteeing unconditional stability and conservation of the invariant $\sum |u_k|^2$.

**Coordinate Transformation (Identity Basis):**
To match the conservation laws of the 1:1 ODE system, we use the identity basis (no normalization factors):
$$u_1 = A_1, \quad u_2 = A_2, \quad u_3 = A_3$$

**Update Rule:**
$$\mathbf{u}_{n+1} = \left( \mathbf{I} - \frac{1}{2}\mathbf{\Omega}_n \right)^{-1} \left( \mathbf{I} + \frac{1}{2}\mathbf{\Omega}_n \right) \mathbf{u}_n$$

**The Interaction Matrix $\mathbf{\Omega}_n$:**
$\mathbf{\Omega}_n$ is constructed to be **Skew-Hermitian** ($\mathbf{\Omega}^\dagger = -\mathbf{\Omega}$) using the geometric mean of the frozen couplings.

$$
\mathbf{\Omega}_n = i \begin{pmatrix} 
0 & \mu_{12} & \mu_{13} \\
\mu_{12}^* & 0 & \mu_{23} \\
\mu_{13}^* & \mu_{23}^* & 0
\end{pmatrix}
$$

**Effective Coupling Elements (Frozen at $z_n$):**
Using the Fourier coefficients $\beta(\Delta k) = \int_z^{z+h} \kappa e^{i\Delta k \xi} d\xi$:

* **SHG ($u_1 \leftrightarrow u_2$):**
    $$\mu_{12} = \beta_{SHG}(\Delta k_{SHG}) \cdot A_1^*(z_n)$$

* **SFG ($u_1 \leftrightarrow u_3$):**
    $$\mu_{13} = \beta_{SFG}(\Delta k_{SFG}) \cdot A_2^*(z_n)$$

* **SFG ($u_2 \leftrightarrow u_3$):**
    $$\mu_{23} = 2 \cdot \beta_{SFG}(\Delta k_{SFG}) \cdot A_1^*(z_n)$$
    *(Coefficient 2 comes from the ODE)*

**Computational Cost:**
* Requires one $3 \times 3$ analytic linear solve per step.
* Allows step size $h$ to be increased by factor of $10\text{-}50\times$ compared to Euler.
