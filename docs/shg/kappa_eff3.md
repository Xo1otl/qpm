# Overview
Derivation of **Robust Generalized LECC** via **Parametric Mapping**. Decouples shape/phase at local scale to eliminate SVEA constraints; valid for discontinuous $\delta(z)$ and $K(z)$.

# 1. Parametric Mapping
Define structure via "Local Unit Cell" parameters at position $z$.
* Local coordinate: $\xi = z' - z$ where $\xi \in [-\Lambda_0/2, \Lambda_0/2]$
* Local profile: $\kappa_{\text{loc}}(\xi; z) = K(\xi - \delta(z))$
* $\delta(z)$: Shift parameter (can be discontinuous)

# 2. Local Fourier Projection
Extract $m$-th Fourier component $\kappa_m$ of local cell. Parameters $\delta(z)$ treated as constants during $\xi$-integration (Separation of Scales).
$$
\kappa_m(z) = \frac{1}{\Lambda_0} \int_{-\Lambda_0/2}^{\Lambda_0/2} K(\xi - \delta(z)) e^{i G_m \xi} d\xi
$$

Apply Shift Theorem (valid for fixed $\delta$):
$$
\begin{aligned}
\kappa_m(z) &= \frac{1}{\Lambda_0} e^{i G_m \delta(z)} \int_{-\Lambda_0/2}^{\Lambda_0/2} K(\xi') e^{i G_m \xi'} d\xi' \\
&= \tilde{K}_m e^{i G_m \delta(z)}
\end{aligned}
$$

# 3. Global Continuum Limit
Reconstruct total field via Riemann Summation of local coefficients over length $L$.
$$
\sum_{n} \kappa_m(z_n) e^{i \Delta \beta z_n} \Lambda_0 \xrightarrow{\Lambda_0 \to 0} \int_0^L \kappa_{\text{eff}}(z) e^{i \Delta \beta z} dz
$$

**Resulting Effective Nonlinearity:**
$$
\kappa_{\text{eff}}(z) = \underbrace{\tilde{K}_m}_{\text{Shape (Amp)}} \cdot \underbrace{e^{i G_m \delta(z)}}_{\text{Position (Phase)}}
$$
