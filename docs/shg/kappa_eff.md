# Overview
Derivation of **Generalized Effective Nonlinearity**. Defines relationship between arbitrary local shape $S_n$ and effective nonlinearity $\kappa_{\text{eff}}$.

# 1. Baseline Expression
$$
A_2(L) = i \kappa_{\text{mat}} A_1^2 \int_0^L s(z) e^{i \Delta k z} dz
$$

# 2. Discrete Decomposition
$$
\int_0^L (\dots) dz \approx \sum_{n=0}^{N-1} e^{i \Delta k z_n} \int_{-\Lambda_0/2}^{\Lambda_0/2} s_n(\xi) e^{i \Delta k \xi} d\xi
$$

**Approximations:**
1.  $\Delta k \approx G_m = 2\pi m / \Lambda_0$
2.  $e^{i \Delta k z_n} = (-1)^m e^{i \Delta \beta z_n}$

**Unit Cell Integral:**
Let $s_n(\xi) = S_n(\xi - \delta_n)$.
$$
\int_{-\Lambda_0/2}^{\Lambda_0/2} s_n(\xi) e^{i G_m \xi} d\xi = e^{i G_m \delta_n} \underbrace{\int_{-\infty}^{\infty} S_n(\xi) e^{i G_m \xi} d\xi}_{\mathcal{F}[S_n](-G_m)}
$$

# 3. Continuum Limit
$$
A_2(L) \approx i \kappa_{\text{mat}} A_1^2 \frac{1}{\Lambda_0} \sum_{n=0}^{N-1} \left[ (-1)^m \mathcal{F}[S_n](-G_m) e^{i G_m \delta_n} \right] e^{i \Delta \beta z_n} \Lambda_0 \approx i \frac{\kappa_{\text{mat}}}{\Lambda_0} A_1^2 \int_0^L \left[ (-1)^m \mathcal{F}[S(z)](-G_m) e^{i G_m \delta(z)} \right] e^{i \Delta \beta z} dz
$$

# 4. Generalized $\kappa_{\text{eff}}$
$$
\kappa_{\text{eff}}(z) = i (-1)^m \frac{\kappa_{\text{mat}}}{\Lambda_0} e^{i G_m \delta(z)} \mathcal{F}[S(z)](-G_m)
$$
**Components:**
* **Phase:** $\delta(z)$
* **Amplitude:** $\mathcal{F}[S(z)](-G_m)$

# 5. Inverse Design
Target: $\kappa_{\text{target}}(z) = A_{\text{target}}(z) e^{i \Phi_{\text{target}}(z)}$

**Step 1: Position (Phase)**
$$
\delta(z) = \frac{1}{G_m} \left( \Phi_{\text{target}}(z) - \frac{\pi}{2} - m\pi - \arg(\mathcal{F}[S(z)](-G_m)) \right)
$$

**Step 2: Profile (Amplitude)**
$$
|\mathcal{F}[S(z)](-G_m)| = \frac{\Lambda_0 A_{\text{target}}(z)}{\kappa_{\text{mat}}}
$$

# Key Concerns
The amplitude term should be explicitly written as a function of $z$. 
The current notation $\mathcal{F}[S(z)](-G_m)$ looks constant, but the shape $S$ changes along the propagation axis. 

# Task
Correct formulation.
