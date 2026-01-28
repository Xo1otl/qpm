# Overview
Derivation of **Fejer’s Method** (FM-QPM) via **Fourier Decomposition** of nonlinear coefficient $d(z)$. Control of **Phase** (via Local Period/Frequency) and **Amplitude** (via Duty Cycle) under **Slowly Varying** approximation.

# 1. Integral Solution
Undepleted pump approximation ($A_1 \approx \text{const}$). SHG output amplitude $A_2(L)$ defined by transfer function integral:
$$
A_2(L) = -i \gamma A_1^2 \int_{-\infty}^{+\infty} d(z) e^{-i \Delta k z} dz
$$
* $\Delta k$: Mismatch wavevector.
* $d(z)$: Spatially varying nonlinear coefficient.

# 2. Fourier Decomposition
Approximation: $d(z)$ is **slowly varying**. Decomposed into spatial Fourier components $d_m(z)$:
$$
d(z) = \sum_{m} d_m(z)
$$
Factor out linear phase component $K_{0m}z$ (carrier):
$$
d_m(z) = |d_m(z)| \exp[i K_{0m} z + i \Phi_m(z)]
$$
* $K_{0m}$: Nominal grating vector.
* $\Phi_m(z)$: Slowly varying phase deviation.

# 3. Frequency Modulation (Phase)
Phase $\Phi_m(z)$ derived from **Local K-vector** $K_m(z)$ (Frequency Modulation):
$$
K_m(z) = K_{0m} + \frac{d\Phi_m}{dz} = \frac{2\pi m}{\Lambda(z)}
$$
Approximation: **Adiabatic evolution**. Variations in period $\Lambda(z)$ must be slow to define local K-vector.
$$
\Phi_m(z) = \int_0^z \left( K_m(\xi) - K_{0m} \right) d\xi
$$

# 4. Amplitude Modulation (Duty Cycle)
Amplitude $|d_m(z)|$ controlled by **Local Duty Cycle** $G(z)$ for $m$-th order:
$$
|d_m(z)| = \frac{2}{\pi m} d_{\text{eff}} \sin(\pi m G(z))
$$
**Approximation: $G(z)$ varies slowly to maintain valid Fourier decomposition.**

# 5. Effective Nonlinearity
Substitute single dominant Fourier order ($K_{0m} \approx \Delta k$) into integral.
$$
A_2(L) \approx -i \gamma A_1^2 \int_0^L \underbrace{\left[ |d_m(z)| e^{i \Phi_m(z)} \right]}_{\kappa_{\text{eff}}(z)} e^{-i (\Delta k - K_{0m}) z} dz
$$
Resulting $\kappa_{\text{eff}}(z)$:
$$
\kappa_{\text{eff}}(z) = \underbrace{\frac{2 d_{\text{eff}}}{\pi m} \sin(\pi m G(z))}_{\text{AM (Duty Cycle)}} \times \underbrace{\exp\left( i \int_0^z \left[ \frac{2\pi m}{\Lambda(\xi)} - \frac{2\pi m}{\Lambda_0} \right] d\xi \right)}_{\text{FM (Period Integral)}}
$$
