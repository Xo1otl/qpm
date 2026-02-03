# Overview
Derivation of **LECC Reformulation**.
* **Method A (Limited):** Sliding Window (Requires slowly varying $\delta$).
* **Method B (Robust):** Parametric Mapping (Valid for discontinuous $\delta$).

# 1. Baseline Expression
Standard QPM integral for SHG (continuum limit).
$$A_2(L) = i A_1^2 \int_0^L \kappa_{\text{eff}}(z) e^{i \Delta \beta z} dz$$
*(Note: Using $\Delta \beta = \Delta k - G_m$ for QPM context)*

# 2. LECC Definition (The Bifurcation)

### Method A: Sliding Window (Old)
Integration of coupling coefficient over a **sliding** period window $[z, z+\Lambda_0]$.
$$\mathcal{F}[\kappa]_{\text{slide}}(z) = \int_{z}^{z+\Lambda_0} \kappa(z') e^{i \Delta k z'} dz'$$
> **Constraint:** Window overlaps cause averaging. $\delta(z)$ must be continuous.

### Method B: Parametric Mapping (New/Robust)
Definition via a **static local unit cell** at point $z$. No sliding overlap.
$$\kappa_{\text{loc}}(z) = \frac{1}{\Lambda_0} \int_{-\Lambda_0/2}^{\Lambda_0/2} \kappa(\xi; z) e^{-i G_m \xi} d\xi$$
> **Improvement:** Decouples local structure ($G_m$) from beam propagation ($\Delta k$). Allows step-function $\delta(z)$.

# 3. Fourier Shift Derivation

### Method A (Old)
Substitute $\kappa(z') = K(\xi - \delta)$ into sliding integral.
Assume $\delta$ is constant over integration (SVEA).
$$
\begin{aligned}
\text{Result} &\propto \tilde{K}(-\Delta k) e^{i \Delta k \delta(z)}
\end{aligned}
$$
> **Flaw:** Mixing $\Delta k$ into the shift term makes phase control dependent on laser wavelength.

### Method B (New/Robust)
Apply Shift Theorem to the local cell integral.
Shift $\delta(z)$ is structurally defined relative to grid $G_m$.
$$
\begin{aligned}
\kappa_{\text{loc}}(z) &= \frac{1}{\Lambda_0} \int K(\xi - \delta(z)) e^{-i G_m \xi} d\xi \\
&= \underbrace{\left[ \frac{1}{\Lambda_0} \tilde{K}(G_m) \right]}_{\text{Shape}} \cdot \underbrace{e^{-i G_m \delta(z)}}_{\text{Position}}
\end{aligned}
$$
> **Fix:** Shift term depends on $G_m$, making it a pure structural property.

# 4. Resulting Effective Nonlinearity

### Comparison
| Feature | Method A (Old) | **Method B (Robust)** |
| :--- | :--- | :--- |
| **Formula** | $\kappa_{\text{eff}} \propto \tilde{K}(-\Delta k) e^{i \Delta k \delta}$ | $\kappa_{\text{eff}} \propto \tilde{K}(G_m) e^{-i G_m \delta}$ |
| **Variable** | $\Delta k$ (Laser dependent) | $G_m$ (Structure dependent) |
| **Discontinuity** | Invalid (Smearing) | **Valid (Exact)** |

# 5. Robust Inverse Design
Use **Method B** for precise control.

**1. Amplitude Control (Duty Cycle)**
Target the Fourier coefficient of the shape at the grid frequency $G_m$.
$$|\tilde{K}(G_m)| = \text{Target Amplitude}$$

**2. Phase Control (Pulse Position)**
Map the target phase directly to the structural shift $\delta$.
$$ -G_m \cdot \delta(z) + \arg(\tilde{K}) = \text{Target Phase}(z)$$
