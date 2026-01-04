# Overview
Derivation from **CWEs** via **Integral Solution** to **Discrete Summation**. Evaluating **Structure Factor** and applying **The Continuum Limit** yields **Effective Nonlinearity**.

# 1. CWEs
$$
\frac{dA_1}{dz} = i \kappa(z)^* A_1 A_2 e^{-i \Delta k z}
$$
$$
\frac{dA_2}{dz} = i \kappa(z) A_1^2 e^{i \Delta k z}
$$
* $\Delta k = k_{2\omega} - 2k_{\omega}$

# 2. Integral Solution
Undepleted Pump ($A_1(z) \approx A_1(0)$).  
Model: $\kappa(z) = \kappa_{\text{mat}} \cdot g(z)$, $g(z) \in \{+1, -1\}$.

$$
A_2(L) = i \kappa_{\text{mat}} A_1^2 \int_0^L g(z) e^{i \Delta k z} dz
$$

# 3. Discrete Summation
$N$ periods, length $\Lambda_0$.
$$
\int_0^L \dots = \sum_{n=0}^{N-1} \int_{z_n - \Lambda_0/2}^{z_n + \Lambda_0/2} g_n(z) e^{i \Delta k z} dz
$$
* $z_n = n\Lambda_0 + \Lambda_0/2$
* $\xi = z - z_n$
$$
e^{i \Delta k z} = e^{i \Delta k z_n} \cdot e^{i \Delta k \xi}
$$
$$
A_2(L) = i \kappa_{\text{mat}} A_1^2 \sum_{n=0}^{N-1} e^{i \Delta k z_n} \underbrace{\left[ \int_{-\Lambda_0/2}^{\Lambda_0/2} g_n(\xi) e^{i \Delta k \xi} d\xi \right]}_{I_{\text{unit}}(n)}
$$

# 4. Structure Factor
QPM $\Delta k \approx G_m = 2\pi m / \Lambda_0$.
$$
I_{\text{unit}}(n) = -\int_{-\Lambda_0/2}^{-w_n/2} e^{i G_m \xi} d\xi + \int_{-w_n/2}^{+w_n/2} e^{i G_m \xi} d\xi - \int_{+w_n/2}^{+\Lambda_0/2} e^{i G_m \xi} d\xi
$$

Pulse ($w_n = D_n \Lambda_0$):
$$
\int_{-w_n/2}^{+w_n/2} e^{i G_m \xi} d\xi = \frac{2}{G_m} \sin\left(\frac{G_m w_n}{2}\right)
$$
Gap sum vanishes. Total:
$$
I_{\text{unit}}(n) = \Lambda_0 \frac{2}{m \pi} \sin(m \pi D_n)
$$

# 5. The Continuum Limit
Riemann Sum $\Lambda_0 \to dz$:
$$A_2(L) = i \kappa_{\text{mat}} A_1^2 \sum_{n=0}^{N-1} e^{i \Delta k z_n} \cdot \left( \Lambda_0 \frac{2}{m \pi} \sin(m \pi D_n) \right)$$
$$\sum_{n=0}^{N-1} (\dots) \Lambda_0 \xrightarrow{\Lambda_0 \to 0} \int_0^L (\dots) dz$$
$$A_2(L) \approx i \kappa_{\text{mat}} A_1^2 \int_0^L \frac{2}{m \pi} \sin(m \pi D(z)) e^{i \Delta k z} dz$$

# 6. Effective Nonlinearity
Comparison to generic integral:
$$
A_2(L) = i A_1^2 \int_0^L \kappa_{\text{eff}}(z) e^{i \Delta k z} dz
$$
Result:
$$
\kappa_{\text{eff}}(z) = \kappa_{\text{mat}} \frac{2}{m \pi} \sin(m \pi D(z))
$$