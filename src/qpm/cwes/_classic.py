# import numpy as np

# TODO: Implement a classical coupled-wave equation simulation to verify the results of simulate_twm
r"""
### **Coupled-Wave Equations**

Third-Harmonic Generation (THG) through cascaded second-order nonlinear processes (SHG+SFG) is described by the following system of coupled-wave equations.

$$
\begin{align}
\frac{d A_1}{dz} &= i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right] \tag{1} \\
\frac{d A_2}{dz} &= i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \tag {2} \\
\frac{d A_3}{dz} &= i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right] \tag{3}
\end{align}
$$

Here, $\boldsymbol{A}(z)$ represents the complex amplitude vector for each wave, $\kappa(z)$ is the coupling coefficient, and $\Delta k_j$ is the phase mismatch.
"""  # noqa: E501
