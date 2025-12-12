## Coupled ODEs for Cascaded THG

#### **1. Coupled-Wave Equations**

Third-harmonic generation (THG) through cascaded second-order nonlinear processes (SHG+SFG) is described by the following system of coupled-wave equations.

$$\frac{d A_1}{dz} = i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right] \\ \frac{d A_2}{dz} = i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \\ \frac{d A_3}{dz} = i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right]$$

Here, $\boldsymbol{A}(z)$ is the complex amplitude vector for each wave, $\kappa(z)$ is the coupling coefficient, and $\Delta k_j$ is the phase mismatch.

Since the amplitude $A_j$ is defined such that "$|A_j|^2 \propto$ light intensity," the energy conservation law is given by the simple sum $\sum |A_j|^2 = I_{const}$.

---

#### **2. Canonical Form**

The $z$-dependence of the equations is removed by a canonical transformation.

$$\boldsymbol{B}(z) = e^{i\boldsymbol{L}z} \boldsymbol{A}(z) \quad \text{where} \quad \boldsymbol{L} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & \Delta k_1 & 0 \\ 0 & 0 & \Delta k_1 + \Delta k_2 \end{pmatrix}$$

The equation of motion that $\boldsymbol{B}$ follows is described in the following canonical form using a Hamiltonian $K$.

$$\frac{d\boldsymbol{B}}{dz} = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K(\boldsymbol{B}, \boldsymbol{B}^*) \quad \text{where} \quad \boldsymbol{J} = \text{diag}(1, 2, 3)$$

The specific form of the Hamiltonian is given by:

$$K(\boldsymbol{B}, \boldsymbol{B}^*) = \underbrace{\frac{\kappa(z)}{2} \left( B_1^2 B_2^* + (B_1^*)^2 B_2 \right) + \kappa(z) \left( B_1 B_2 B_3^* + B_1^* B_2^* B_3 \right)}_{K_{NL}} + \underbrace{\frac{\Delta k_1}{2} |B_2|^2 + \frac{\Delta k_1 + \Delta k_2}{3} |B_3|^2}_{K_{LIN}}$$

In aperiodically poled structure, to compensate for the phase mismatch $\Delta k$, the sign of the nonlinear coefficient $\kappa(z)$ is inverted with a period $\Lambda \approx 2\pi/\Delta k$. While $\kappa(z)$ has a stiffness close to $\Delta k$ for the entire system, it is constant within each domain.

---

#### **3. Integral Form**

By splitting the Hamiltonian $K$ into a nonlinear term generator $K_{NL}$ and a linear term generator $K_{LIN}$, the equation of motion can be rewritten as follows:
$$\frac{d\boldsymbol{B}}{dz} = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} (K_{LIN} + K_{NL}) = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{LIN} + i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{NL}$$

Here, calculating each term, we get:
$$i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{LIN} = i \begin{pmatrix} 0 & 0 & 0 \\ 0 & \Delta k_1 & 0 \\ 0 & 0 & \Delta k_1 + \Delta k_2 \end{pmatrix} \begin{pmatrix} B_1 \\ B_2 \\ B_3 \end{pmatrix} \equiv i \boldsymbol{L} \boldsymbol{B}$$
$$i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} K_{NL} = i \kappa(z) \begin{pmatrix} B_1^* B_2 + B_2^* B_3 \\ B_1^2 + 2 B_1^* B_3 \\ 3 B_1 B_2 \end{pmatrix} \equiv i \boldsymbol{N}(\boldsymbol{B}, \boldsymbol{B}^*)$$

Thus, the equation of motion can be expressed in a form where the linear and nonlinear terms are separated.
$$\frac{d\boldsymbol{B}}{dz} = i \left( \boldsymbol{L}\boldsymbol{B} + \boldsymbol{N}(\boldsymbol{B}, \boldsymbol{B}^*) \right)$$

The exact solution to this equation is as follows:

$$\boldsymbol{B}(z_n+h) = e^{i\boldsymbol{L}h} \boldsymbol{B}(z_n) + i \int_0^h e^{i\boldsymbol{L}(h-\tau')} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau'), \boldsymbol{B}^*(z_n+\tau')) d\tau'$$

---

#### **4. Analytical Solution Using an Approximation**

We find an analytical solution for a single domain using a perturbative approach. The linear solution, $`\boldsymbol{B}(z_n + \tau')=e^{i\boldsymbol{L}\tau'}\boldsymbol{B}(z_n)`$, is substituted into the nonlinear term, allowing the integral to be solved analytically. This allows the entire domain width to be treated as a single integration step.

1.  **Definition of the Integral Function:**
    Using the domain parameter $P_n$, the integral function $\phi$ is defined as follows.
    $$
    \phi(\omega, h) = \begin{cases}
    \frac{e^{i\omega h} - 1}{i\omega} & (\omega \neq 0) \\
    h & (\omega = 0)
    \end{cases}
    $$

2.  **Calculation of the Nonlinear Evolution Term:**
    The state change vector due to nonlinear effects within a step, $\Delta \boldsymbol{B}_{NL}$, is calculated using the following analytical solution.

$$B_{NL, 1} = i\kappa_n e^{il_1 h_n} \left[ B_{1n}^* B_{2n} \phi(l_2-l_1-l_1, h_n) + B_{2n}^* B_{3n} \phi(l_3-l_2-l_1, h_n) \right] \\ B_{NL, 2} = i\kappa_n e^{il_2 h_n} \left[ B_{1n}^2 \phi(2l_1-l_2, h_n) + 2 B_{1n}^* B_{3n} \phi(l_3-l_1-l_2, h_n) \right] \\ B_{NL, 3} = i \, 3\kappa_n e^{il_3 h_n} \left[ B_{1n} B_{2n} \phi(l_1+l_2-l_3, h_n) \right]$$

3.  **State Prediction:**
    By combining linear and nonlinear evolution, the state vector at the end of the step, $\boldsymbol{B}_{pred}$, is calculated.

$$\boldsymbol{B}_{pred} = e^{i\boldsymbol{l}h_n} \boldsymbol{B}_n + \boldsymbol{B}_{NL}$$

# Task
More exact form
$$
\frac{d A_1}{dz} = i \left[ \kappa_{SHG}(z) A_2 A_1^* e^{i\Delta k_{SHG} z} + \kappa_{SFG}(z) A_3 A_2^* e^{i\Delta k_{SFG} z} \right]
$$

$$
\frac{d A_2}{dz} = i \left[ \kappa_{SHG}(z) A_1^2 e^{-i\Delta k_{SHG} z} + 2 \kappa_{SFG}(z) A_3 A_1^* e^{i\Delta k_{SFG} z} \right]
$$

$$
\frac{d A_3}{dz} = i \left[ 3 \kappa_{SFG}(z) A_1 A_2 e^{-i\Delta k_{SFG} z} \right]
$$

Can this CWEs also be calculated using the same technique?