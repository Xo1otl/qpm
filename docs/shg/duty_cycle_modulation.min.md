# Overview
Derivation of **Complex Effective Nonlinearity** via **Shifted Pulse Model** in QPM. Control of **Amplitude** (via Duty Cycle) and **Phase** (via Pulse Position) yields arbitrary $\kappa_{\text{eff}}(z)$.

# 1. Shifted Pulse Model
$N$ periods, length $\Lambda_0$. Pulse center shifted by $\delta_n$.
* $n$-th domain center: $z_n = n\Lambda_0 + \Lambda_0/2$
* Pulse center: $z_c = z_n + \delta_n$
* Pulse range: $[z_c - w_n/2, z_c + w_n/2]$

# 2. Integral Solution
Undepleted Pump ($A_1(z) \approx A_1(0)$).
$$
A_2(L) = i \kappa_{\text{mat}} A_1^2 \int_0^L g(z) e^{i \Delta k z} dz
$$

# 3. Discrete Summation
Period-wise decomposition.
$$
A_2(L) = i \kappa_{\text{mat}} A_1^2 \sum_{n=0}^{N-1} e^{i \Delta k z_n} \underbrace{\left[ \int_{-\Lambda_0/2}^{\Lambda_0/2} g_n(\xi) e^{i \Delta k \xi} d\xi \right]}_{I_{\text{unit}}(n)}
$$
* $\xi = z - z_n$

# 4. Structure Factor
QPM $\Delta k \approx G_m$. Background $+1$, Pulse $-1$ (Shift $\delta_n$, Width $w_n$).
$$
I_{\text{unit}}(n) = \int_{-\Lambda_0/2}^{\Lambda_0/2} (+1) e^{i G_m \xi} d\xi - 2 \int_{\delta_n - w_n/2}^{\delta_n + w_n/2} (+1) e^{i G_m \xi} d\xi
$$

Background vanishes ($m \neq 0$). Pulse term:
$$
\int_{\delta_n - w_n/2}^{\delta_n + w_n/2} e^{i G_m \xi} d\xi = \left[ \frac{e^{i G_m \xi}}{i G_m} \right]_{\delta_n - w_n/2}^{\delta_n + w_n/2} = e^{i G_m \delta_n} \frac{2}{G_m} \sin\left(\frac{G_m w_n}{2}\right)
$$
Total:
$$
I_{\text{unit}}(n) = - \Lambda_0 \frac{2}{m \pi} \sin(m \pi D_n) e^{i G_m \delta_n}
$$
* $D_n = w_n / \Lambda_0$

# 5. The Continuum Limit
Riemann Sum $\Lambda_0 \to dz$:
$$
\sum_{n=0}^{N-1} (\dots) \Lambda_0 \xrightarrow{\Lambda_0 \to 0} \int_0^L (\dots) dz
$$
$$
A_2(L) \approx i \kappa_{\text{mat}} A_1^2 \int_0^L \left[ - \frac{2}{m \pi} \sin(m \pi D(z)) e^{i G_m \delta(z)} \right] e^{i \Delta k z} dz
$$

# 6. Complex Effective Nonlinearity
Comparison to generic integral:
$$
\kappa_{\text{eff}}(z) = - \kappa_{\text{mat}} \frac{2}{m \pi} \sin(m \pi D(z)) e^{i G_m \delta(z)}
$$
