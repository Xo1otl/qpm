# Overview
**InitialState** -> **TimeEvolution** -> **IndexConstruction** -> **ModeSolving** -> **CouplingCalculation**

# InitialState
Rectangular step function (lower crystal).
* **Dimensions**: Depth $d_{PE} = 2\sqrt{D_{PE} t_{PE}}$, Width $W$.
* **Profile** ($t=0$):
    $$C(x,y) = \begin{cases} C_0 & \text{if } 0 \le x \le d_{PE} \text{ AND } |y| \le W/2 \\ 0 & \text{otherwise} \end{cases}$$

# TimeEvolution
**2D Diffusion Equation** (Buried).
$$\frac{\partial C}{\partial t} = \frac{\partial}{\partial x}\left(D_x \frac{\partial C}{\partial x}\right) + \frac{\partial}{\partial y}\left(D_y \frac{\partial C}{\partial y}\right)$$
* **Domain**: $x \in [-L_x, L_x]$, $y \in [-L_y, L_y]$.
* **BC**: Continuity at $x=0$, Dirichlet at edges.
* **Coefficients**: $D_x, D_y$.

# IndexConstruction
Z-cut MgO:SLT ($n_e \parallel x$).
$$n(x,y) = n_{\text{sub}} + \Delta n_0 \frac{C(x,y)}{C_0}$$

# ModeSolving
TM modes ($E_x$ dominant).
* **Fields**: $E_{x,\omega}$ (Pump), $E_{x,2\omega}$ (SHG).

# CouplingCalculation
Effective nonlinearity $d_{eff} = \frac{2}{\pi} d_{33}$ (first-order QPM).
Scalar field $E_{\omega} \equiv \vec{E}_{\omega} \cdot \hat{x}$.

### 1. SHG ($\omega + \omega \to 2\omega$)
$$\kappa_{\text{SHG}} = \frac{\omega_{2}\epsilon_0}{4} d_{eff} \iint [E_{x, 2\omega}]^* [E_{x, \omega}]^2 \, dx dy$$

### 2. SFG ($\omega + 2\omega \to 3\omega$)
$$\kappa_{\text{SFG}} = \frac{\omega_3\epsilon_0}{2} d_{eff} \iint [E_{x, 3\omega}]^* E_{x, 2\omega} E_{x, \omega} \, dx dy$$

# Parameters
| Parameter | Symbol | Value | Unit |
| :--- | :--- | :--- | :--- |
| PE Mask Width | $W$ | $48$ | $\mu\text{m}$ |
| PE Time | $t_{PE}$ | $8$ | h |
| PE Diff. Coeff. | $D_{PE}$ | $0.045$ | $\mu\text{m}^2/\text{h}$ |
| Annealing Time | $t_{anneal}$ | $90$ | h |
| Diff. Coeff. (Depth) | $D_x$ | $1.3$ | $\mu\text{m}^2/\text{h}$ |
| Diff. Coeff. (Width) | $D_y$ | $1.3/1.5$ | $\mu\text{m}^2/\text{h}$ |
| Sim Limit (Depth) | $L_x$ | $50$ | $\mu\text{m}$ |
| Sim Limit (Width) | $L_y$ | $50$ | $\mu\text{m}$ |
| Temp | $T$ | $25.5$ | $^\circ\text{C}$ |
| Nonlinear Coeff. | $d_{33}$ | $13.8$ | pm/V |
| Pump Wavelength | $\lambda_{FW}$ | $1064$ | nm |
| SHG Wavelength | $\lambda_{SHW}$ | $532$ | nm |
| Max Index Change ($\omega$) | $\Delta n_0(\omega)$ | $0.012$ | - |
| Max Index Change ($2\omega$) | $\Delta n_0(2\omega)$ | $0.017$ | - |
