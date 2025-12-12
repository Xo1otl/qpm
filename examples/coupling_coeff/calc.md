# Overview
This document summarizes the method for calculating the coupling coefficients $\kappa_{SHG}$ and $\kappa_{SFG}$ for MgO:SLT waveguides, based on the provided context.

**Methodology**:
1.  **H+ Concentration**: Solve diffusion equations to determine proton distribution.
2.  **Refractive Index**: Convert proton concentration to refractive index change.
3.  **Mode Analysis**: Calculate electric field distributions ($E$) for relevant frequencies.
4.  **Coupling Coefficient**: Compute overlap integrals $\kappa$.

# H+ Concentration
The proton concentration $C(x,y,t)$ is determined by solving the diffusion equation.

**1D Model (Slab)**:
During Proton Exchange (PE):
$$
\frac{\partial C}{\partial t} = \frac{\partial}{\partial x}\left(D_{PE}\frac{\partial C}{\partial x}\right)
$$
Solution (Step profile approximation): penetration depth $d_{PE} = 2\sqrt{D_{PE} t_{PE}}$.

After Annealing (APE):
$$
C(x) = \frac{A h_{PE}}{2} \left[ \text{erf}\left(\frac{d_{PE} + x}{d_a}\right) + \text{erf}\left(\frac{d_{PE} - x}{d_a}\right) \right]
$$
where $d_a = 2\sqrt{D_a t_a}$.

**2D Model (Channel)**:
$$
\frac{\partial C}{\partial t} = \frac{\partial}{\partial x}\left(D_x \frac{\partial C}{\partial x}\right) + \frac{\partial}{\partial y}\left(D_y \frac{\partial C}{\partial y}\right)
$$
*Assumption*: $D_x$ (depth, Z-axis) is different from $D_y$ (width, X/Y-axis). Context suggests $D_x \approx 1.5 D_{lateral}$? (Check context: "Da in Z axis is 1.5 times Da in X,Y axis").
*Boundary Conditions*: $D=0$ at crystal-air interface.

**Parameters**:
-   $D_{PE}$ (230°C): $0.045 \, \mu m^2/h$ (measured).
-   $D_a$ (400°C): $1.3 \, \mu m^2/h$ (measured).

# Refractive Index
The refractive index change $\Delta n$ is proportional to the H+ concentration.

$$
\Delta n(x,y) = \Delta n_0 \frac{C(x,y)}{C_0}
$$
-   $\Delta n_0$: Refractive index increase in the initial PE region.
    -   @ 1030 nm: $\Delta n_0 \approx 0.012$
    -   @ 532 nm: $\Delta n_0 \approx 0.017$ (implied from context)
-   Total index: $n(x,y) = n_{sub} + \Delta n(x,y)$
-   $n_{sub}$: Substrate index (derived from Sellmeier equation).

# Mode Analysis
Solve the wave equation using the calculated $n(x,y)$ to find the eigenmodes (electric fields).
-   Method: Finite Element Method (e.g., FemSIM).
-   Outputs:
    -   $E_\omega(x,y)$ (Fundamental)
    -   $E_{2\omega}(x,y)$ (SHG)
    -   $E_{3\omega}(x,y)$ (SFG)

# Coupling Coefficient
Calculate $\kappa$ using the overlap integrals.
Assumption: $d(x,y) = d_{33}$ (constant).
Value: $d_{33} = 1.38 \times 10^{-5} \, \mu m/V$.

**SHG**:
$$
\kappa_{SHG} = \frac{2\omega\epsilon_0 d_{33}}{4} \iint E_{2\omega}(x,y)^* E_\omega(x,y)^2 dxdy
$$

**SFG**:
$$
\kappa_{SFG} = \frac{\omega_3\epsilon_0 d_{33}}{2} \iint E_3(x,y)^* E_2(x,y) E_1(x,y) dxdy
$$
