# Derivation of SPDC Joint Spectral Amplitude (JSA)

### I. Hamiltonian

The interaction Hamiltonian describes the nonlinear coupling in the crystal. Based on the source text, the process is mediated by the second-order nonlinearity $\chi^{(2)}$We express the Hamiltonian density integrated over the crystal volume:

$$
\hat{H}_I(t) = C \left[ \iint dx dy \, \chi^{(2)} f_p(x,y)f_s^*(x,y)f_i^*(x,y) \right] \int_{-\infty}^{\infty} dz \, s_{nl}(z) \, E_p^{(+)}(z,t) \hat{E}_s^{(-)}(z,t) \hat{E}_i^{(-)}(z,t) + h.c.
$$

* $f_k(x,y)$: Transverse spatial mode profiles, representing the nonlinear overlap.
* $s_{nl}(z)$: Structure of the nonlinear domains (e.g., periodic or aperiodic poling).
* $C$: Constant absorbing vacuum permittivity and normalization factors.

---

### II. Field Expansions

To satisfy energy conservation, the positive frequency component of the classical pump ($E^{(+)}$) and the negative frequency components of the quantized signal/idler fields ($\hat{E}^{(-)}$, associated with creation operators) must follow consistent phase definitions.

**Pump Field (Classical):**
$$
E_p^{(+)}(z,t) = \int d\omega_p \, \alpha(\omega_p) \, e^{i(\beta_p(\omega_p)z - \omega_p t)}
$$
* $\alpha(\omega_p)$: Pump spectral envelope function.

**Signal and Idler Fields (Quantized):**
The time dependence is set to $+i\omega t$ for the creation operator term to ensure energy conservation.
$$
\hat{E}_{s,i}^{(-)}(z,t) \propto \int d\omega_{s,i} \, \hat{a}_{s,i}^\dagger(\omega_{s,i}) \, e^{-i(\beta_{s,i}(\omega_{s,i})z - \omega_{s,i} t)}
$$
* $\hat{a}^\dagger$: Creation operator for signal/idler photons.

---

### III. Time Evolution

We treat the interaction as a perturbation. The state evolves from the vacuum state $|0\rangle$ using first-order perturbation theory:

$$
|\Psi\rangle \approx |0\rangle - \frac{i}{\hbar} \int_{-\infty}^{\infty} dt \, \hat{H}_I(t) |0\rangle
$$

Substituting the field expansions into the Hamiltonian:

$$
|\Psi\rangle \propto \iiint d\omega_p d\omega_s d\omega_i \, \alpha(\omega_p) \hat{a}_s^\dagger(\omega_s) \hat{a}_i^\dagger(\omega_i) \times \mathcal{T}(\omega) \times \mathcal{L}(\omega) |00\rangle
$$

---

### IV. Conservation Laws

The integration over time and space yields the conservation laws for energy and momentum.

**Energy Conservation (Time Integral):**
$$
\mathcal{T}(\omega) = \int_{-\infty}^{\infty} dt \, e^{-i(\omega_p - \omega_s - \omega_i)t} = 2\pi \delta(\omega_p - \omega_s - \omega_i)
$$
* This forces $\omega_p = \omega_s + \omega_i$.

**Momentum Conservation / Phase Matching (Space Integral):**
$$
\mathcal{L}(\omega) = \int_{-\infty}^{\infty} dz \, s_{nl}(z) \, e^{i(\beta_p - \beta_s - \beta_i)z} = \Phi(\Delta\beta)
$$
* $\Phi(\Delta\beta)$ is the phase-matching function, defined as the Fourier transform of the nonlinearity structure $s_{nl}(z)$.

---

### V. Joint Spectral Amplitude (JSA)

Applying the delta function to integrate over $\omega_p$, we arrive at the final two-photon state. The spectral properties are entirely determined by the product of the pump envelope and the phase-matching function.

$$
|\Psi\rangle = \iint d\omega_s d\omega_i \, \underbrace{\left[ \alpha(\omega_s + \omega_i) \times \Phi(\Delta \beta) \right]}_{\text{JSA}(\omega_s, \omega_i)} \, \hat{a}_s^\dagger(\omega_s) \hat{a}_i^\dagger(\omega_i) |00\rangle
$$

$$
\text{JSA}(\omega_s, \omega_i) = \alpha(\omega_s + \omega_i) \times \Phi(\beta_p(\omega_s+\omega_i) - \beta_s(\omega_s) - \beta_i(\omega_i))
$$

* **Pump Envelope Function $\alpha(\omega_s + \omega_i)$:** Determines the anti-diagonal width of the JSA (energy conservation).
* **Phase-Matching Function $\Phi(\Delta\beta)$:** Determines the diagonal width and structure (momentum conservation), which can be engineered by manipulating $s_{nl}(z)$.

### Necessity of Gaussian Spectrum for Pure State Generation

The purity of a single photon heralded from an SPDC source is determined by the spectral correlations between the signal and idler photons. Mathematically, the two-photon state is described by the Joint Spectral Amplitude (JSA), $f(\omega_s, \omega_i)$.

For the heralded photon to be in a pure state, the JSA must be factorable (separable) into independent signal and idler components:

$$
f(\omega_s, \omega_i) = \psi_s(\omega_s) \times \psi_i(\omega_i)
$$

If the JSA cannot be separated in this way, detection of one photon projects the other into a mixed state, degrading purity.

#### The Role of Gaussian Functions
The JSA is the product of two functions:
1.  **Pump Envelope Function ($\alpha$):** Defined by the laser pulse, typically Gaussian: $\alpha(\omega_s + \omega_i) \approx \exp[-(\omega_s + \omega_i)^2 / \sigma_p^2]$.
2.  **Phase-Matching Function ($\Phi$):** Defined by the crystal nonlinearity. For standard periodic poling, this is a `sinc` function ($\sin(x)/x$).

**The Problem:** The product of a Gaussian (pump) and a Sinc function (crystal) is generally **not separable** in the coordinates $\omega_s$ and $\omega_i$, leading to spectral entanglement.

**The Solution:** The product of two Gaussians *is* another Gaussian, which can often be factored. Therefore, to ensure separability, the phase-matching function $\Phi$ must also be engineered to be Gaussian:

$$
\Phi(\Delta k) \propto \exp(-\gamma \Delta k^2)
$$

This Gaussian phase-matching function eliminates the "side lobes" present in sinc functions that cause spectral correlations, allowing the JSA to be factorable under specific group-velocity matching conditions.