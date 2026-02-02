# Overview
Derivation of **Generalized Effective Nonlinearity**. Defines relationship between parametric local shape $S(\xi; z)$ and effective nonlinearity $\kappa_{\text{eff}}$.

# 1. Baseline Expression
$$
A_2(L) = i \kappa_{\text{mat}} A_1^2 \int_0^L s(z) e^{i \Delta k z} dz
$$

# 2. Discrete Decomposition
$$
\int_0^L (\dots) dz \approx \sum_{n=0}^{N-1} e^{i \Delta k z_n} \int_{-\Lambda_0/2}^{\Lambda_0/2} s_n(\xi) e^{i \Delta k \xi} d\xi
$$

**Approx:**
1. $\Delta k \approx G_m = 2\pi m / \Lambda_0$
2. $e^{i \Delta k z_n} = (-1)^m e^{i \Delta \beta z_n}$

**Parametric Shape Definition:**
$$
s_n(\xi) = S(\xi - \delta_n; z_n)
$$
$$
\tilde{S}(k; z) \equiv \mathcal{F}_{\xi} [S(\xi; z)](k) = \int_{-\infty}^{\infty} S(\xi; z) e^{-i k \xi} d\xi
$$

# 3. Continuum Limit
$$
A_2(L) \approx i \frac{\kappa_{\text{mat}}}{\Lambda_0} A_1^2 \int_0^L \left[ (-1)^m \tilde{S}(-G_m; z) e^{i G_m \delta(z)} \right] e^{i \Delta \beta z} dz
$$

# 4. Generalized $\kappa_{\text{eff}}$
$$
\kappa_{\text{eff}}(z) = i (-1)^m \frac{\kappa_{\text{mat}}}{\Lambda_0} e^{i G_m \delta(z)} \tilde{S}(-G_m; z)
$$

**Polar Form:**
$$
\tilde{S}(-G_m; z) = \left| \tilde{S}(-G_m; z) \right| e^{i \psi_S(z)}
$$

# 5. Inverse Design
**Target:**
$$
\kappa_{\text{target}}(z) = A_{\text{target}}(z) e^{i \Phi_{\text{target}}(z)}
$$

**Step 1: Profile (Amplitude Control)**
$$
\left| \tilde{S}(-G_m; z) \right| = \frac{\Lambda_0 A_{\text{target}}(z)}{\kappa_{\text{mat}}}
$$

**Step 2: Position (Phase Control)**
$$
\delta(z) = \frac{1}{G_m} \left( \Phi_{\text{target}}(z) - \frac{\pi}{2} - m\pi - \psi_S(z) \right)
$$
