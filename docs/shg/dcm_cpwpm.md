# Hybrid Chirped-Pulse QPM (CPM)

## 1. Model Definitions
We define a quasi-phase matching (QPM) structure with spatially varying period $\Lambda(z)$, duty cycle $D(z)$, and pulse position shift $\delta(z)$.

* **Base Grid Phase (FM):** The cumulative phase of the variable grating vector $K_g(z) = 2\pi m / \Lambda(z)$.
  $$
  \Phi_g(z) = \int_0^z \frac{2\pi m}{\Lambda(\xi)} d\xi
  $$
* **Domain Centers:** Defined by integer crossings of the base phase.
  $$
  \Phi_g(z_n) = 2\pi n, \quad n \in \mathbb{Z}
  $$
* **Pulse Position (PPM):** The center of the $n$-th inverted domain is shifted by $\delta_n$ relative to the grid.
  $$
  z_{c,n} = z_n + \delta(z_n)
  $$

## 2. Complex Effective Nonlinearity
Under the slowly varying approximation ($\Lambda'(z) \ll 1$ and $\delta'(z) \ll 1$), the discrete summation converges to the integral of a complex effective nonlinearity $\kappa_{\text{eff}}(z)$.

$$
A_2(L) \approx -i \gamma A_1^2 \int_0^L \kappa_{\text{eff}}(z) e^{-i \Delta k z} dz
$$

$$
\kappa_{\text{eff}}(z) = \underbrace{\frac{2 d_{\text{eff}}}{m \pi} \sin(m \pi D(z))}_{\text{Amplitude (Apodization)}} \cdot \exp\left[ i \underbrace{\left( \int_0^z \frac{2\pi m}{\Lambda(\xi)} d\xi + \frac{2\pi m \delta(z)}{\Lambda(z)} \right)}_{\Psi(z): \text{Total QPM Phase}} \right]
$$

## 3. Generalized Stationary Phase Condition
Optimal conversion occurs where the local gradient of the QPM phase matches the wavevector mismatch $\Delta k$.

$$
\frac{d\Psi}{dz} = \Delta k(z)
$$

Expanding the derivative yields the **Load Balancing Equation**:

$$
\Delta k(z) = \underbrace{\frac{2\pi m}{\Lambda(z)}}_{\text{Coarse Chirp (FM)}} + \underbrace{2\pi m \left( \frac{\delta'(z)}{\Lambda(z)} - \frac{\delta(z)\Lambda'(z)}{\Lambda^2(z)} \right)}_{\text{Fine Phase Correction (PPM)}}
$$

## 4. Validity Criteria
To ensure the Riemann sum approximation remains valid (avoiding the "narrow sinc" violation), the modulation must be adiabatic:

1. **Adiabatic Period:** $|\Lambda'(z)| \ll 1$
2. **Adiabatic Shift:** $|\delta'(z)| \ll 1$
