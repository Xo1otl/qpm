# Overview
**NDPA Derivation** for **CWEs**.

# Coupled-Wave Equations (NDPA)
Approximation: $A_1(z) \approx A_1(0) = const$.
$$\begin{aligned}
\frac{d A_2}{dz} &= i \kappa_{SHG}(z) A_1^2 e^{-i\Delta k_{SHG} z} \\
\frac{d A_3}{dz} &= i \, 3\kappa_{SFG}(z) A_1 A_2 e^{-i\Delta k_{SFG} z}
\end{aligned}$$

# Derivation
**Step 1**: Solve for $A_2(z)$ ($A_2(0)=0$).
$$A_2(z) = i A_1^2 \int_0^{z} \kappa_{SHG}(z_1) e^{-i\Delta k_{SHG} z_1} \, dz_1$$

**Step 2**: Substitute into $A_3$ equation.
$$\frac{dA_3}{dz} = i \, 3\kappa_{SFG}(z) A_1 \left( i A_1^2 \int_0^{z} \kappa_{SHG}(z_1) e^{-i\Delta k_{SHG} z_1} \, dz_1 \right) e^{-i\Delta k_{SFG} z}$$
$$\frac{dA_3}{dz} = -3 A_1^3 \kappa_{SFG}(z) e^{-i\Delta k_{SFG} z} \int_0^{z} \kappa_{SHG}(z_1) e^{-i\Delta k_{SHG} z_1} \, dz_1$$

**Step 3**: Integrate $0 \to L$.
$$A_3(L) = -3 A_1^3 \int_0^L dz_2 \, \kappa_{SFG}(z_2) e^{-i\Delta k_{SFG} z_2} \left( \int_0^{z_2} dz_1 \, \kappa_{SHG}(z_1) e^{-i\Delta k_{SHG} z_1} \right)$$

# Result
$$A_3(L) = -3 A_1^3 S$$
$$S \equiv \int_0^L dz_2 \int_0^{z_2} dz_1 \, \kappa_{SFG}(z_2) \kappa_{SHG}(z_1) e^{-i(\Delta k_{SHG} z_1 + \Delta k_{SFG} z_2)}$$
