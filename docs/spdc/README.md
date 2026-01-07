1. **Biphoton State Representation & Pure State Condition**
   $$ |\Psi\rangle = \iint d\omega_s \, d\omega_i \, f(\omega_s, \omega_i) \, \hat{a}^\dagger_s(\omega_s) \, \hat{a}^\dagger_i(\omega_i) \, |0\rangle $$
   $$ \text{Tr}_i(|\Psi\rangle\langle\Psi|) = \text{Pure} \iff f(\omega_s, \omega_i) = \phi(\omega_s)\psi(\omega_i) \quad [\text{Factorizable}] $$

2. **Definition of JSA**
   $$ f(\omega_s, \omega_i) = \alpha(\omega_s + \omega_i) \times \Phi(\Delta k) $$
   $$ \qquad \qquad [\text{Pump: Gaussian}] \qquad [\text{PMF}] $$

3. **Definition of PMF**
   $$ \Phi(\Delta k) = \mathcal{F}[ d(z) ] $$

4. **Logical Consequence**  
   To achieve a pure state $\iff$ $[\text{Factorizable}]$
   $$ \downarrow \quad (\text{Since the Pump is Gaussian...}) $$
   The PMF must also be $[\text{Gaussian}]$
   $$ \downarrow \quad (\text{Due to binary material constraints...}) $$
   $d(z)$ is Duty-Cycle Modulated to simulate a Gaussian profile
