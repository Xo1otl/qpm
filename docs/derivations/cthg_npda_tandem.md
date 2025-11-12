### 1. The Functional

$$
S = \int_0^L dz_2 \, K_2(z_2) \left( \int_0^{z_2} dz_1 \, K_1(z_1) \right)
$$
Where:
* $K_1(z) = \kappa(z) e^{-i\Delta k_1 z}$ (SHG term)
* $K_2(z) = \kappa(z) e^{-i\Delta k_2 z}$ (SFG term)

### 2. Rotating-Wave Approximation (RWA)

The tandem structure and RWA imply:

| Region | $z$ | $K_1(z)$ (SHG-Matched) | $K_2(z)$ (SFG-Matched) |
| :--- | :--- | :--- | :--- |
| **Section 1** | $[0, L_1]$ | $\bar{\kappa}$ | $0$ |
| **Section 2** | $[L_1, L]$ | $0$ | $\bar{\kappa}$ |

### 3. Evaluating the Integral

Split $S$ at $L_1$:
$$
S = \int_0^{L_1} dz_2 \, K_2(z_2) \left( \int_0^{z_2} dz_1 \, K_1(z_1) \right) + \int_{L_1}^L dz_2 \, K_2(z_2) \left( \int_0^{z_2} dz_1 \, K_1(z_1) \right)
$$

**Part 1 ($S_1$): $z_2 \in [0, L_1]$**
Using RWA, $K_2(z_2) = 0$ in this region.
$$
S_1 = \int_0^{L_1} dz_2 \, (0) \left( \dots \right) = 0
$$

**Part 2 ($S_2$): $z_2 \in [L_1, L]$**
Using RWA, $K_2(z_2) = \bar{\kappa}$ in this region.
$$
S_2 = \int_{L_1}^L dz_2 \, (\bar{\kappa}) \left( \int_0^{z_2} dz_1 \, K_1(z_1) \right)
$$
Evaluate the inner integral $\int_0^{z_2} dz_1$. Since $z_2 > L_1$, split at $L_1$:
$$
\int_0^{z_2} dz_1 \, K_1(z_1) = \int_0^{L_1} dz_1 \, K_1(z_1) + \int_{L_1}^{z_2} dz_1 \, K_1(z_1)
$$
Apply RWA to the inner integral:
$$
\int_0^{z_2} dz_1 \, K_1(z_1) = \int_0^{L_1} (\bar{\kappa}) \, dz_1 + \int_{L_1}^{z_2} (0) \, dz_1 = \bar{\kappa} L_1
$$
Substitute this constant result back into $S_2$:
$$
S_2 = \int_{L_1}^L dz_2 \, (\bar{\kappa}) \left( \bar{\kappa} L_1 \right)
= \bar{\kappa}^2 L_1 \int_{L_1}^L dz_2
= \bar{\kappa}^2 L_1 [z_2]_{L_1}^L
= \bar{\kappa}^2 L_1 (L - L_1)
$$

**Total Functional**
$$
S = S_1 + S_2 = \bar{\kappa}^2 L_1 (L - L_1)
$$

### 4. Maximizing Efficiency

Efficiency $\eta \propto |S|^2$:
$$
\eta \propto [L_1 (L - L_1)]^2
$$
Maximize $f(L_1) = L_1 (L - L_1) = L L_1 - L_1^2$.
Find the critical point:
$$
\frac{df}{dL_1} = \frac{d}{dL_1} (L L_1 - L_1^2) = L - 2L_1
$$
Set derivative to zero:
$$
L - 2L_1 = 0
$$
$$
L_1 = \frac{L}{2}
$$
The ratio of SHG length ($L_1$) to SFG length ($L - L_1$) is 1:1.
