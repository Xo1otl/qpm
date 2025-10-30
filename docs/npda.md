### **Coupled-Wave Equations**

Third-Harmonic Generation (THG) through a cascaded second-order nonlinear process (SHG+SFG) is described by the following system of coupled-wave equations.

$$
\begin{align}
\frac{d A_1}{dz} &= i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right] \tag{1} \\
\frac{d A_2}{dz} &= i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \tag {2} \\
\frac{d A_3}{dz} &= i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right] \tag{3}
\end{align}
$$

Where $A_j(z)$ is the complex amplitude of each wave, $\kappa(z)$ is the coupling coefficient, and $\Delta k_j$ represents the phase mismatch.

# Under Approximation
Under the undepleted pump approximation ($A_1(z) \approx A_1$) and assuming low conversion efficiency to the third harmonic ($|A_1| \gg |A_3(z)|$) the system of equations simplifies significantly by defining a two-dimensional "cascaded susceptibility function", $S(\Delta k_1, \Delta k_2)$:

$$
S(\Delta k_1, \Delta k_2) = \int_0^L dz_2 \int_0^{z_2} dz_1 \, \kappa(z_1) \kappa(z_2) e^{-i(\Delta k_1 z_1 + \Delta k_2 z_2)}
$$

The final third-harmonic amplitude is then directly proportional to this function.

$$
A_3(L) = -3 A_1^3 \, S(\Delta k_1, \Delta k_2)
$$
