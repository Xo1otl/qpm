# Overview
**Exact Discrete Splitting** of the **Shifted Pulse Model**. **Midpoint Riemann Sum** yields valid **Continuum Limit** (Approx $\Delta k \approx G_m$ justified).

# Shifted Pulse Model (Exact)
$N$ periods, length $\Lambda_0$. Shift $\delta_n$ defined by local phase $\phi_n$.
* $n$-th domain center: $z_n = n\Lambda_0 + \Lambda_0/2$
* Shift scaling: $\delta_n = \frac{\phi_n}{2\pi m} \Lambda_0$
* Pulse range: $[z_n + \delta_n - w_n/2, \quad z_n + \delta_n + w_n/2]$

# Discrete Summation (Exact)
Splitting is exact. Aliasing relation $e^{i \Delta k z_n} = (-1)^m e^{i \Delta \beta z_n}$ is **Exact** by definition ($\Delta k = G_m + \Delta \beta$).
$$
A_2(L) = i \kappa_{\text{mat}} A_1^2 \sum_{n=0}^{N-1} (-1)^m e^{i \Delta \beta z_n} I_n(\Delta k)
$$

# Unit Cell Integral (Approx $\Delta k \to G_m$)
Approximate $I_n(\Delta k) \approx I_n(G_m)$ inside the integral. Error vanishes in limit over $L$.
$$
I_n(G_m) = \underbrace{\int_{-\Lambda_0/2}^{\Lambda_0/2} (+1) e^{i G_m \xi} d\xi}_{\text{bg}=0 \ (m \neq 0)} - 2 \underbrace{\int_{\delta_n - w_n/2}^{\delta_n + w_n/2} (+1) e^{i G_m \xi} d\xi}_{\text{pulse}}
$$

Pulse term:
$$
\int_{\delta_n - w_n/2}^{\delta_n + w_n/2} e^{i G_m \xi} d\xi = e^{i \phi_n} \frac{2}{G_m} \sin(m \pi D_n)
$$

Total Result:
$$
I_{\text{unit}}(n) = - \Lambda_0 \frac{2}{m \pi} \sin(m \pi D_n) e^{i \phi_n}
$$

# Continuum Limit (Rigorous)
Sum becomes **Midpoint Riemann Sum** for samples $f(z_n)$.
$$
A_2(L) \approx \sum_{n=0}^{N-1} \underbrace{\left[ i \kappa_{\text{mat}} A_1^2 (-1)^{m+1} \frac{2}{m \pi} \sin(m \pi D_n) e^{i \phi_n} e^{i \Delta \beta z_n} \right]}_{f(z_n)} \Lambda_0
$$
As $\Lambda_0 \to 0$, sum converges to $\int_0^L f(z) dz$.

# Complex Effective Nonlinearity
$$
\kappa_{\text{eff}}(z) = (-1)^{m+1} \kappa_{\text{mat}} \frac{2}{m \pi} \sin(m \pi D(z)) e^{i \phi(z)}
$$
