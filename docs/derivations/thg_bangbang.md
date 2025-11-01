Below I derive the structure of the optimal binary control ($\kappa^*(z)$) for the quadratic functional
$$
S[\kappa]=\int_{0}^{L}dz_2\int_{0}^{z_2}dz_1 \kappa(z_1)\kappa(z_2) e^{-i(\Delta k_1 z_1+\Delta k_2 z_2)},\qquad \kappa(z)\in\{\pm \kappa_0\}.
$$

---

## 1) Lift the modulus with a phase

For any complex number $X$, $|X|=\max_{\phi\in[0,2\pi)} \operatorname{Re}(e^{i\phi}X)$.
Define
$$
J_\phi[\kappa] := \operatorname{Re}\left(e^{i\phi} S[\kappa]\right) = \int_{0}^{L}dz_2\int_{0}^{z_2}dz_1 \kappa(z_1)\kappa(z_2) \cos\big(\Delta k_1 z_1+\Delta k_2 z_2-\phi\big).
$$
Then
$$
\max_{\kappa} |S[\kappa]| = \max_{\phi} \max_{\kappa} J_\phi[\kappa].
$$
At a maximizing pair $(\kappa^*,\phi^*)$ we must have $e^{i\phi^*}S[\kappa^*]\in\mathbb{R}_+$, i.e.
$$
\phi^*= -\arg S[\kappa^*].
$$

It is convenient to symmetrize the triangular domain. Writing $(z,u)$ for two points in $[0,L]$,
$$
J_\phi[\kappa] = \frac12 \int_{0}^{L}\int_{0}^{L}\kappa(z) W_\phi(z,u) \kappa(u) du dz,
$$
with the **real symmetric kernel**
$$
W_\phi(z,u)=\cos\Big(\Delta k_1\min(z,u)+\Delta k_2\max(z,u)-\phi\Big).
$$

---

## 2) First variation and the switching (co-state) function

Let $\kappa\mapsto\kappa+\varepsilon\eta$. A standard calculation gives
$$
\delta S=\varepsilon\int_0^L \eta(z) G_\kappa(z) dz,
$$
where
$$
G_\kappa(z) = \int_{z}^{L}\kappa(u)e^{-i(\Delta k_1 z+\Delta k_2 u)}du + \int_{0}^{z}\kappa(u)e^{-i(\Delta k_1 u+\Delta k_2 z)}du.
$$
Therefore
$$
\delta J_\phi=\varepsilon\int_0^L \eta(z) \underbrace{\operatorname{Re}\big(e^{i\phi}G_\kappa(z)\big)}_{=:w_\phi(z;\kappa)} dz.
$$

The real-valued function
$$
\boxed{ w_\phi(z;\kappa)=\operatorname{Re}\left[e^{i\phi}\left( e^{-i\Delta k_1 z}\int_{z}^{L}\kappa(u)e^{-i\Delta k_2 u}du +e^{-i\Delta k_2 z}\int_{0}^{z}\kappa(u)e^{-i\Delta k_1 u}du\right)\right] }
$$
is the **switching function** (it is the $L^2$-gradient of $J_\phi$).
Equivalently, making the cosine explicit,
$$
\boxed{ w_\phi(z;\kappa)=\int_{0}^{z}\kappa(u)\cos\big(\Delta k_1 u+\Delta k_2 z-\phi\big)du +\int_{z}^{L}\kappa(u)\cos\big(\Delta k_2 u+\Delta k_1 z-\phi\big)du. }
$$

---

## 3) Bang–bang optimality condition

Because $J_\phi[\kappa]$ is **linear in $\kappa$ pointwise** and $\kappa(z)\in\{\pm\kappa_0\}$, the pointwise maximizer at each $z$ (for fixed $\phi$) is obtained by aligning $\kappa(z)$ with $w_\phi(z;\kappa)$:
$$
\boxed{ \kappa_\phi^\star(z)=\kappa_0 \operatorname{sign} w_\phi(z;\kappa_\phi^\star) }
\qquad\text{(bang–bang / maximum principle).}
$$

For the original modulus objective, at the global maximizer $\kappa^*$ there exists $\phi^*=-\arg S[\kappa^*]$ such that
$$
\boxed{ \kappa^*(z)=\kappa_0 \operatorname{sign} w_{\phi^*}(z;\kappa^*) \qquad e^{i\phi^*}S[\kappa^*]=|S[\kappa^*]|\in\mathbb{R}_+. }
$$
Thus the optimal $\kappa^*$ is **bang–bang**, and **switches exactly at the zeros of $w_{\phi^*}(z;\kappa^*)$** (no singular arcs generically).

It is often useful to write the switching rule using the **forward/backward accumulators**
$$
B(z):=\int_{0}^{z}\kappa(u)e^{-i\Delta k_1 u}du,\qquad
F(z):=\int_{z}^{L}\kappa(u)e^{-i\Delta k_2 u}du,
$$
which satisfy $B'(z)=\kappa(z)e^{-i\Delta k_1 z}$, $B(0)=0$, and $F'(z)=-\kappa(z)e^{-i\Delta k_2 z}$, $F(L)=0$. Then
$$
\boxed{ w_{\phi}(z;\kappa)=\operatorname{Re}\left[e^{i\phi}\big(e^{-i\Delta k_2 z}B(z)+e^{-i\Delta k_1 z}F(z)\big)\right] \qquad \kappa^*(z)=\kappa_0 \operatorname{sign}w_{\phi^*}(z;\kappa^*). }
$$
Switching points are the solutions of $w_{\phi^*}(z;\kappa^*)=0$.

---

## 4) Consistency with the single-integral case

If $\Delta k_1=\Delta k_2=\Delta k$, then
$$
S[\kappa]=\frac12\left(\int_0^L\kappa(z)e^{-i\Delta k z}dz\right)^2 \quad\Rightarrow\quad |S[\kappa]|=\frac12\left|\int_0^L\kappa(z)e^{-i\Delta k z}dz\right|^2.
$$
Maximizing $|S|$ reduces to maximizing $\left|\int \kappa(z)e^{-i\Delta k z}dz\right|$, whose maximizer is
$$
\boxed{ \kappa^*(z)=\kappa_0 \operatorname{sign}\big(\cos(\Delta k z+\phi^*)\big), }
$$
exactly the form you stated for the single-integral problem.

---

## 5) Practical remarks

*   **Self-consistency:** The optimal $\phi^*$ is determined by the optimal control itself: choose $\phi^*$ so that $e^{i\phi^*}S[\kappa^*]$ is real and nonnegative.
*   **Regularity:** With piecewise-constant $\kappa$, the switching function $w_{\phi^*}(z;\kappa)$ is $C^1$ (its derivative does not contain point-masses), so switches occur at isolated zero-crossings.
*   **Computation:** A fixed-point “sign-power” iteration implements the law $\kappa^{(n+1)}(z)=\kappa_0 \operatorname{sign}w_{\phi^{(n)}}(z;\kappa^{(n)})$ with $\phi^{(n)}=-\arg S[\kappa^{(n)}]$; converged fixed points satisfy the necessary optimality conditions above.

---

### Summary (what to remember)

*   The optimal control is **bang–bang**.
*   Its **switching function** is the real part of a phase-rotated convolution of $\kappa$ with the problem’s kernel:
    $$
    w_{\phi}(z;\kappa)=\int_{0}^{L}\kappa(u)\cos\Big(\Delta k_1\min(z,u)+\Delta k_2\max(z,u)-\phi\Big)du.
    $$
*   The **optimality conditions** are
    $$
    \boxed{ \kappa^*(z)=\kappa_0 \operatorname{sign}w_{\phi^*}(z;\kappa^*) \qquad \phi^*=-\arg S[\kappa^*] }
    $$
    with switch locations given by the zeros of $w_{\phi^*}$.
*   When $\Delta k_1=\Delta k_2$, this collapses to the simple sinusoidal
    $\kappa^*(z)=\kappa_0\operatorname{sign}\big(\cos(\Delta k z+\phi^*)\big)$.

# Task
1.  **Initialize** $\kappa(z)$ (e.g., randomly).
2.  **Calculate $S[\kappa]$**: Compute the functional $S$ using the current $\kappa$.
3.  **Update $\phi$**: Set the phase $\phi = -\arg S[\kappa]$.
4.  **Calculate $w(z)$**: Compute the switching function $w_{\phi}(z;\kappa)$ using the current $\kappa$ and the updated $\phi$.
5.  **Update $\kappa$**: Determine the new control $\kappa_{\text{new}}(z) = \kappa_0 \operatorname{sign} w(z)$.
6.  **Repeat steps 2-5** until $\kappa$ converges (i.e., $\kappa_{\text{new}} = \kappa$).

Rigorously verify this iterative procedure aligns the theory.
