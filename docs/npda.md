## Coupled-Wave Equations

Third-Harmonic Generation (THG) through a cascaded second-order nonlinear process (SHG+SFG) is described by the following system of coupled-wave equations.

$$
\begin{align}
\frac{d A_1}{dz} &= i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right] \tag{1} \\
\frac{d A_2}{dz} &= i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \tag {2} \\
\frac{d A_3}{dz} &= i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right] \tag{3}
\end{align}
$$

Where $A_j(z)$ is the complex amplitude of each wave, $\kappa(z)$ is the coupling coefficient, and $\Delta k_j$ represents the phase mismatch.

## Under Approximation
Under the undepleted pump approximation ($A_1(z) \approx A_1$) and assuming low conversion efficiency to the third harmonic ($|A_1| \gg |A_3(z)|$), the system's behavior is captured by a **Second-Order Term of a Dyson Series**, $S$. This functional takes the coupling profile $\kappa(z)$ as input and yields a value dependent on the phase mismatches $\Delta k_1$ and $\Delta k_2$:

$$
S[\kappa](\Delta k_1, \Delta k_2) = \int_0^L dz_2 \int_0^{z_2} dz_1 \, \kappa(z_1) \kappa(z_2) e^{-i(\Delta k_1 z_1 + \Delta k_2 z_2)}
$$

The final third-harmonic amplitude is then directly proportional to this function.

$$
A_3(L) = -3 A_1^3 \, S(\Delta k_1, \Delta k_2)
$$

## QPM discretization and closed-form pieces

Let $0=z_0<z_1<\dots<z_N=L$ with domains $[z_{n-1},z_n]$, lengths $\ell_n=z_n-z_{n-1}$, and piecewise-constant $\kappa(z)=\kappa_n$ on each domain.

Define
$$
R([a,b];k)=\int_a^b e^{-ik z}dz=\frac{e^{-ika}-e^{-ikb}}{ik}
=e^{-ik\frac{a+b}{2}}\ell\,\mathrm{sinc}\left(\frac{k\ell}{2}\right),\quad \ell=b-a
$$
and abbreviate $R_n(k)=R([z_{n-1},z_n];k)$.

For the triangular (same-domain) part,
$$
J([a,b];\Delta k_1,\Delta k_2)
=\int_a^b e^{-i\Delta k_2 z_2}\left(\int_a^{z_2}e^{-i\Delta k_1 z_1}dz_1\right)dz_2
=\frac{e^{-i\Delta k_1 a}R([a,b];\Delta k_2)-R([a,b];\Delta k_1+\Delta k_2)}{i\Delta k_1}
$$
with continuous limits $J([a,b];0,\Delta k_2) = \int_a^b e^{-i\Delta k_2 z}(z-a)dz$ and $J([a,b];0,0)=\ell^2/2$.
Write $J_n(\Delta k_1,\Delta k_2)=J([z_{n-1},z_n];\Delta k_1,\Delta k_2)$.

## Discretized $S$ and $O(N)$ evaluation

For a piecewise-constant coupling profile $\kappa(z)$, the functional can be evaluated by splitting the integral over the domains:
$$
S[\kappa](\Delta k_1,\Delta k_2)
=\sum_{n=1}^{N}\sum_{m=1}^{n-1}\kappa_m\kappa_n R_m(\Delta k_1)R_n(\Delta k_2)
+\sum_{n=1}^{N}\kappa_n^2 J_n(\Delta k_1,\Delta k_2).
\tag{*}
$$

Let $s^{(1)}_n=\kappa_n R_n(\Delta k_1)$ and $s^{(2)}_n=\kappa_n R_n(\Delta k_2)$. Then the double sum is
$$
\sum_{n=1}^{N} s^{(2)}_n\left(\sum_{m=1}^{n-1} s^{(1)}_m\right)
$$

The inner sum is a cumulative sum. This structure allows the entire double summation to be calculated in a single pass, achieving $O(N)$ time complexity. Consequently, the total susceptibility $S$ can be computed efficiently.
