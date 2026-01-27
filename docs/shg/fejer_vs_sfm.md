### 1. Fixed-Lattice Phase Modulation (User's Method)
**Paradigm:** Discrete perturbations on a rigid periodic grid.
* **Independent Variable:** Fixed reference period $\Lambda_0$.
* **Design Variables:**
    * **Positional Shift ($\delta_n$):** Displacement of the $n$-th domain center relative to the grid ($z_n = n\Lambda_0 + \delta_n$).
    * **Pulse Width ($w_n$):** Controls amplitude $|A_n|$.
* **Mechanism:** Encodes phase $\Phi(z)$ via positional detuning $\delta_n = \Phi(z_n) / G_m$.

### 2. Adiabatic Frequency Modulation (Fejér’s Method)
**Paradigm:** Continuous modulation of local periodicity under the Slowly Varying Envelope Approximation (SVEA).
* **Independent Variable:** Spatially varying local period $\Lambda(z)$.
* **Design Variables:**
    * [cite_start]**Local Period ($\Lambda(z)$):** Defined differentially as $\Lambda(z) = 2\pi / (K_0 + d\Phi/dz)$[cite: 31, 33, 41].
    * [cite_start]**Local Duty Cycle ($D(z)$):** Controls amplitude $|A(z)|$[cite: 31, 36].
* **Mechanism:** Encodes phase via the derivative of the phase function (Frequency Modulation). The domain center is implicitly fixed to the center of the local period.

### 3. Unconstrained Discrete Mapping (Integrated Method)
**Paradigm:** Global quantization of the accumulated phase integral.
* **Independent Variable:** None (Grid-free).
* **Design Variables:**
    * **Absolute Position ($z_n$):** The exact coordinate satisfying the constructive interference condition.
    * **Absolute Width ($w_n$):** The integration limit required for specific field amplitude.
* **Mechanism:** Direct root-finding of the phase quantization equation:
    $$G_m z_n - \Phi_{\text{target}}(z_n) = 2\pi n$$