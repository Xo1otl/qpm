# Overview
**Coupled-Wave Equations** define interaction. **Canonical Form** reveals Hamiltonian structure. **NDPA Derivation** yields 3rd harmonic amplitude via perturbative integration.

# Coupled-Wave Equations
$$\frac{d A_1}{dz} = i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right] \\ \frac{d A_2}{dz} = i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \\ \frac{d A_3}{dz} = i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right]$$
$\boldsymbol{A}$: amplitude ($\sum |A_j|^2 = const$). $\kappa$: coupling. $\Delta k$: mismatch.

# Canonical Form
Rotation removes fast phase:
$$\boldsymbol{B}(z) = e^{i\boldsymbol{L}z} \boldsymbol{A}(z), \quad \boldsymbol{L} = \text{diag}(0, \Delta k_1, \Delta k_1 + \Delta k_2)$$
Hamiltonian dynamics $\dot{\boldsymbol{B}} = i \boldsymbol{J} \nabla_{\boldsymbol{B}^*} (K_{NL} + K_{LIN})$:
$$K = \underbrace{\frac{\kappa}{2} \left( B_1^2 B_2^* + c.c. \right) + \kappa \left( B_1 B_2 B_3^* + c.c. \right)}_{K_{NL}} + \underbrace{\frac{\Delta k_1}{2} |B_2|^2 + \frac{\Delta k_1 + \Delta k_2}{3} |B_3|^2}_{K_{LIN}}$$
QPM: $\kappa(z)$ sign inverted, period $\Lambda \approx 2\pi/\Delta k$.

# NDPA Derivation
**Starting Point**: Full Canonical Equations.
$$\begin{aligned}
\frac{dB_1}{dz}&= i\kappa(z)\big(B_1^*B_2+B_2^*B_3\big) \\
\frac{dB_2}{dz}&= i\Delta k_1 B_2 + i\kappa(z)\big(B_1^2+2B_1^*B_3\big) \\
\frac{dB_3}{dz}&= i(\Delta k_1+\Delta k_2)B_3 + i\,3\kappa(z)B_1B_2
\end{aligned}$$
**Approximation**: NDPA ($B_1 \approx A_1$), linearized.
$$\begin{aligned}
\frac{dB_2}{dz} - i\Delta k_1 B_2 &= i\kappa(z) A_1^2 \\
\frac{dB_3}{dz} - i(\Delta k_1+\Delta k_2) B_3 &= i\,3\kappa(z) A_1 B_2
\end{aligned}$$
**Integration**: Solve for $B_2$.
$$B_2(z) = i A_1^2 e^{i\Delta k_1 z} \int_0^{z} \kappa(z') e^{-i\Delta k_1 z'} \, dz'$$
Substitute into $B_3$ equation and integrate $0 \to L$.
$$A_3(L) = -3A_1^3 S, \quad S \equiv \int_0^L dz_2 \int_0^{z_2} dz_1 \, \kappa(z_2)\kappa(z_1) e^{-i(\Delta k_1 z_1 + \Delta k_2 z_2)}$$