# Overview
**Optimal Length Ratio** for **Accurate Tandem Function**.

# Functional
$$S = \int_0^L dz_2 \, \kappa_{SFG}(z_2) e^{-i\Delta k_{SFG} z_2} \left( \int_0^{z_2} dz_1 \, \kappa_{SHG}(z_1) e^{-i\Delta k_{SHG} z_1} \right)$$

# RWA Model
Two regions with ideal Phase Matching ($\Delta k \to 0$ effectively) and selective coupling.

**Region 1 ($0 \le z \le L_1$)**: SHG active.
*   $\kappa_{SHG} = g_1, \quad \kappa_{SFG} = 0$.

**Region 2 ($L_1 < z \le L$)**: SFG active.
*   $\kappa_{SHG} = 0, \quad \kappa_{SFG} = g_2$.

# Evaluation
Split outer integral at $L_1$.
**Part 1 ($z_2 \le L_1$)**: $\kappa_{SFG}(z_2) = 0 \implies S_1 = 0$.

**Part 2 ($z_2 > L_1$)**: $\kappa_{SFG}(z_2) = g_2$.
$$S = g_2 \int_{L_1}^L dz_2 \left( \int_0^{z_2} \kappa_{SHG}(z_1) dz_1 \right)$$
Inner integral ($z_2 > L_1$):
$$\int_0^{z_2} \kappa_{SHG} dz_1 = \int_0^{L_1} g_1 dz_1 + \int_{L_1}^{z_2} 0 \, dz_1 = g_1 L_1$$
Substitute back:
$$S = g_2 \int_{L_1}^L (g_1 L_1) dz_2 = g_1 g_2 L_1 (L - L_1)$$

# Optimization
Efficiency $\eta \propto |S|^2$. Maximize $f(L_1) = L_1 (L - L_1)$.
$$\frac{df}{dL_1} = L - 2L_1 = 0 \implies L_1 = \frac{L}{2}$$
**Result**: Optimal ratio is **1:1**.
