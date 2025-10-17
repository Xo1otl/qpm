# Introduction

This project provides a differentiable solver for the Coupled-Wave Equations (CWEs) using an improved Exponential Time Differencing (ETD) method. This enables gradient-based optimization for the inverse design of quasi-phase-matching (QPM) structures in nonlinear optical devices.

# Overview

The core components are:

*   **Coupled-Wave Equations (CWEs)**: A model for the interaction of light waves in a nonlinear medium.
*   **Improved ETD Method**: A numerical technique for solving the CWEs.
*   **Differentiable Simulation**: The solver is implemented in JAX, making it differentiable.
*   **Inverse Design via Optimization**: Using the differentiable solver to design and optimize the physical structure of a device.

# Coupled-Wave Equations (CWEs)

The evolution of light wave amplitudes in a nonlinear medium is described by:

$$\frac{d A_1}{dz} = i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right] \\ \frac{d A_2}{dz} = i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \\ \frac{d A_3}{dz} = i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right]$$

# QPM Devices

In QPM devices, the nonlinear coefficient `κ(z)` is spatially modulated to compensate for phase mismatch. This is achieved by creating a structure of discrete domains, where `κ` is constant within each domain but varies between them.

# Improved ETD Method

The CWEs are solved by propagating the fields through this series of discrete domains. For each domain, the output field amplitudes (FW, SHW, THW) are calculated analytically from the input amplitudes using an improved Exponential Time Differencing (ETD) method. The state vector at the end of a single domain is given by:

$$\boldsymbol{B}_{pred} = e^{i\boldsymbol{l}h_n} \boldsymbol{B}_n + \boldsymbol{B}_{NL}$$

The nonlinear term, $\boldsymbol{B}_{NL}$, is calculated analytically. First, an integral function is defined:
$$
\phi(\omega, h) = \begin{cases}
\frac{e^{i\omega h} - 1}{i\omega} & (\omega \neq 0) \\
h & (\omega = 0)
\end{cases}
$$
The components of the nonlinear term $\boldsymbol{B}_{NL} = (B_{NL, 1}, B_{NL, 2}, B_{NL, 3})^T$ are then given by:
$$B_{NL, 1} = i\kappa_n e^{il_1 h_n} \left[ B_{1n}^* B_{2n} \phi(l_2-2l_1, h_n) + B_{2n}^* B_{3n} \phi(l_3-l_2-l_1, h_n) \right] \\ B_{NL, 2} = i\kappa_n e^{il_2 h_n} \left[ B_{1n}^2 \phi(2l_1-l_2, h_n) + 2 B_{1n}^* B_{3n} \phi(l_3-l_1-l_2, h_n) \right] \\ B_{NL, 3} = i \, 3\kappa_n e^{il_3 h_n} \left[ B_{1n} B_{2n} \phi(l_1+l_2-l_3, h_n) \right]$$
where $\boldsymbol{l} = (l_1, l_2, l_3)$ are the phase mismatch terms for each wave.

# Differentiable Simulation

The simulation is implemented in JAX, making the CWEs solver differentiable. This allows for the computation of the gradient of an output parameter (e.g., conversion efficiency) with respect to an input parameter (e.g., the widths of individual domains in a QPM grating).

# Inverse Design via Optimization

The differentiable solver enables inverse design of photonic structures through gradient-based optimization. By defining a loss function, an optimizer iteratively adjusts the physical parameters of the structure. The gradients from the solver guide the optimization process to find a design that minimizes the loss function.
