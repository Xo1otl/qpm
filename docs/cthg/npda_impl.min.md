## Under Approximation
Under the undepleted pump approximation ($A_1(z) \approx A_1$) and assuming low conversion efficiency ($|A_1| \gg |A_3(z)|$), the third-harmonic amplitude is proportional to a double integral functional $S$. This functional depends on the coupling profiles $\kappa_1(z)$ (SHG), $\kappa_2(z)$ (SFG), and phase mismatches $\Delta k_1, \Delta k_2$:

$$
S[\kappa_1, \kappa_2](\Delta k_1, \Delta k_2) = \int_0^L dz_2 \, \kappa_2(z_2) e^{-i\Delta k_2 z_2} \int_0^{z_2} dz_1 \, \kappa_1(z_1) e^{-i\Delta k_1 z_1}
$$

The final third-harmonic amplitude is then directly proportional to this function.

$$
A_3(L) = -3 A_1^3 \, S(\Delta k_1, \Delta k_2)
$$

## QPM discretization and closed-form pieces

Let $0=z_0<z_1<\dots<z_N=L$ with domains $[z_{n-1},z_n]$, lengths $\ell_n=z_n-z_{n-1}$, and piecewise-constant $\kappa_1(z)=\kappa_{1,n}, \kappa_2(z)=\kappa_{2,n}$ on each domain.

Define integrals over the domain $[z_{n-1}, z_n]$ of length $\ell_n$:
$$
R_n(k)=\int_{z_{n-1}}^{z_n} e^{-ik z}dz
=e^{-ik\frac{z_{n-1}+z_n}{2}}\ell_n\,\mathrm{sinc}\left(\frac{k\ell_n}{2\pi}\right)
$$

For the triangular (same-domain) part:
$$
J_n(\Delta k_1,\Delta k_2)
=\int_{z_{n-1}}^{z_n} e^{-i\Delta k_2 z_2}\left(\int_{z_{n-1}}^{z_2}e^{-i\Delta k_1 z_1}dz_1\right)dz_2
=\frac{e^{-i\Delta k_1 z_{n-1}}R_n(\Delta k_2)-R_n(\Delta k_1+\Delta k_2)}{i\Delta k_1}
$$
with continuous limits $J_n(0,\Delta k_2) = \int_{z_{n-1}}^{z_n} e^{-i\Delta k_2 z}(z-z_{n-1})dz$ and $J_n(0,0)=\ell_n^2/2$.

## Discretized $S$ and $O(N)$ evaluation

For piecewise-constant coupling profiles, the functional can be evaluated by splitting the integral over the domains:
$$
S[\kappa_1, \kappa_2](\Delta k_1,\Delta k_2)
=\sum_{n=1}^{N}\sum_{m=1}^{n-1}\kappa_{1,m}\kappa_{2,n} R_m(\Delta k_1)R_n(\Delta k_2)
+\sum_{n=1}^{N}\kappa_{1,n}\kappa_{2,n} J_n(\Delta k_1,\Delta k_2).
\tag{*}
$$

Let $s^{(1)}_n=\kappa_{1,n} R_n(\Delta k_1)$ and $s^{(2)}_n=\kappa_{2,n} R_n(\Delta k_2)$. Then the double sum is
$$
\sum_{n=1}^{N} s^{(2)}_n\left(\sum_{m=1}^{n-1} s^{(1)}_m\right)
$$

The inner sum is a cumulative sum. This structure allows the entire double summation to be calculated in a single pass, achieving $O(N)$ time complexity. Consequently, the total susceptibility $S$ can be computed efficiently.
