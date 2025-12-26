# Overview
Fixing the **Center of Gravity (CG)** prevents phase errors in duty cycle modulation. **Left-Anchored** (2-domain) modulation causes centroid drift, inducing unwanted chirp. **Center-Anchored** (3-domain, symmetric) modulation maintains the centroid, ensuring spectral fidelity.

# Necessity of CG Fixing
*   **Drift**: Varying duty cycle $D$ in `[D*Lp, (1-D)*Lp]` shifts pulse center.
*   **Result**: Introduces position-dependent phase shift $\phi(z)$, breaking Fourier symmetry.
*   **Symptom**: Asymmetric spectra, deviation from target apodization (e.g., Sinc/Gaussian).

# Implementation
*   **Logic**: Symmetrize pulse within period $\Lambda$.
*   **Structure**: 3-domain `[Gap, Pulse, Gap]`.
    *   $w_{gap} = (1 - D_n) \Lambda / 2$
    *   $w_{pulse} = D_n \Lambda$
*   **Mapping**: $D_n = \frac{1}{\pi} \arcsin(|A(z)|)$
*   **Outcome**: Centroid fixed at $\Lambda/2$. Restores pure amplitude modulation.
