# Overview
Deriving the **LECC Reformulation** from the standard **Baseline QPM integral** to establish a framework for Inverse Design.

# Baseline Expression
Standard QPM integral for SHG.
$$A_2(L) = i A_1^2 \int_0^L \kappa(z) e^{i \Delta k z} dz$$

# LECC Definition
Integration of coupling coefficient over local period $\Lambda_0$ targeting the $-\Delta k$ Fourier component to match Baseline.
$$\mathcal{F}[\kappa](-\Delta k) = \int_{z}^{z+\Lambda_0} \kappa(z') e^{i \Delta k z'} dz'$$

# Parametric Profile
Local coordinate $z' = z + \xi$. Profile defined by shape $K$ and shift $\delta$.
$$\kappa(z') = K(\xi - \delta)$$

# Fourier Shift Derivation
Substitute profile into LECC integral. Apply Shift Theorem $\mathcal{F}[f(x-\delta)] = e^{-ik\delta}\mathcal{F}[f]$ where $k = -\Delta k$.
$$
\begin{aligned}
\mathcal{F}[\kappa] &= e^{i \Delta k z} \int_{-\Lambda_0/2}^{\Lambda_0/2} K(\xi - \delta) e^{i \Delta k \xi} d\xi \\
&= e^{i \Delta k z} \left[ e^{-i (-\Delta k) \delta} \tilde{K}(-\Delta k) \right] \\
&= e^{i \Delta k z} \left[ e^{i \Delta k \delta} \tilde{K}(-\Delta k) \right]
\end{aligned}
$$
**Resulting Effective Nonlinearity:**
$$\kappa_{\text{eff}} \propto e^{i \Delta k \delta} \tilde{K}(-\Delta k)$$

# Inverse Design
**1. Amplitude Control (Shape)**
$$|\tilde{K}(-\Delta k)| = \text{Target Amplitude}$$

**2. Phase Control (Position)**
$$\Delta k \cdot \delta + \arg(\tilde{K}) = \text{Target Phase}$$
