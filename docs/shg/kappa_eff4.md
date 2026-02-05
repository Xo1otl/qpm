# Overview
**Generalized Effective Coupling Coefficient** from **Baseline Expression** for Holographic QPM using **Local Fourier Coupling Coefficient**.

# Baseline Expression
Standard QPM integral for SHG.
$$A_2(L) = i A_1^2 \int_0^L \kappa(z) e^{i \Delta k z} dz$$

# LFCC Definition
Integration of nonlinearity over local period $\Lambda_0$ targeting the $-\Delta k$ Fourier component to match Baseline.
$$\mathcal{F}[\kappa](-\Delta k) = \int_{z}^{z+\Lambda_0} \kappa(z') e^{i \Delta k z'} dz'$$

# Parametric Profile
Local coordinate $z' = z + \xi$. Profile defined by shape $K$ and shift $\delta$.

$K(\xi; z)$ is $\Lambda_0$-periodic with respect to $\xi$.

$$\kappa(z') = K(\xi - \delta(z); z)$$

# Fourier Shift Derivation
Substitute profile into LFCC integral. Apply Shift Theorem $\mathcal{F}[f(x-\delta)] = e^{-ik\delta}\mathcal{F}[f]$ where $k = -\Delta k$.
$$
\begin{aligned}
\mathcal{F}[\kappa](-\Delta k) &= e^{i \Delta k z} \mathcal{F}[K(\cdot - \delta(z); z)](-\Delta k) \\
&= e^{i \Delta k z} \left[ e^{-i (-\Delta k) \delta(z)} \mathcal{F}[K(\cdot; z)](-\Delta k) \right] \\
&= e^{i \Delta k z} \left[ e^{i \Delta k \delta(z)} \mathcal{F}[K(\cdot; z)](-\Delta k) \right]
\end{aligned}
$$
**Resulting Effective Nonlinearity:**
$$\kappa_{\text{eff}}(z) = \frac{1}{\Lambda_0} e^{i \Delta k \delta(z)} \mathcal{F}[K(\cdot; z)](-\Delta k)$$

# Inverse Design
**1. Amplitude Control (Shape)**
$$\frac{1}{\Lambda_0} \left| \mathcal{F}[K(\cdot; z)](-\Delta k) \right| = \text{Target Amplitude}(z)$$

**2. Phase Control (Position)**
$$\Delta k \cdot \delta(z) + \arg(\mathcal{F}[K(\cdot; z)](-\Delta k)) = \text{Target Phase}(z)$$
