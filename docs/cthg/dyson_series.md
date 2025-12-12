Below I show (i) how the Hamiltonian form in **etd.md** produces a *linear, inhomogeneous* system under the non‑depleted pump approximation (NDPA), then (ii) how its **Dyson (z‑ordered) expansion** yields, at second order, exactly the NDPA formula in **npda.md**:
$$
A_3(L)=-3A_1^3 S[\kappa](\Delta k_1,\Delta k_2).
$$

## 1) From Hamiltonian to a linear system and its Dyson series

### Canonical equations from (K)

With $\boldsymbol{B}=(B_1,B_2,B_3)^T$, $\boldsymbol{J}=\mathrm{diag}(1,2,3)$, and
$$
K=\frac{\kappa(z)}{2} \left(B_1^2B_2^*+(B_1^*)^2B_2\right)
+\kappa(z) \left(B_1B_2B_3^*+B_1^*B_2^*B_3\right)
+\frac{\Delta k_1}{2}|B_2|^2+\frac{\Delta k_1+\Delta k_2}{3}|B_3|^2,
$$
the canonical flow ($\frac{d\boldsymbol{B}}{dz}=i\boldsymbol{J}\nabla_{\boldsymbol{B}^*}K$) gives
$$
\begin{aligned}
\frac{dB_1}{dz}&= i\kappa(z)\big(B_1^*B_2+B_2^*B_3\big) \\
\frac{dB_2}{dz}&= i\left[\Delta k_1B_2+\kappa(z)B_1^2+2\kappa(z)B_1^*B_3\right] \\
\frac{dB_3}{dz}&= i\left[(\Delta k_1+\Delta k_2)B_3+3\kappa(z)B_1B_2\right].
\end{aligned}
\tag{A}
$$

### NDPA reduction

Under NDPA, take $B_1(z)\approx A_1$ (a constant equal to the input pump amplitude; note $B_1\equiv A_1$ because $L_{11}=0$ in the canonical transform). Then $B_2,B_3$ obey a **linear, inhomogeneous** 2×2 system:
$$
\frac{d}{dz}\begin{pmatrix}B_2\\ B_3\end{pmatrix}
= i\Big(H_0+V(z)\Big)\begin{pmatrix}B_2\\ B_3\end{pmatrix}
+ i\,\mathbf{b}(z),
\qquad
\begin{cases}
H_0=\begin{pmatrix}\Delta k_1&0\\[2pt]0&\Delta k_1+\Delta k_2\end{pmatrix},\\[6pt]
V(z)=\kappa(z)\begin{pmatrix}0&2A_1^*\\[2pt]3A_1&0\end{pmatrix},\\[6pt]
\mathbf{b}(z)=\kappa(z)\begin{pmatrix}A_1^2\\[2pt]0\end{pmatrix}.
\end{cases}
\tag{B}
$$

Let $\mathbf{y}(z)=(B_2(z),B_3(z))^T$. The homogeneous propagator $U(z,z_0)$ solves
$$
\frac{d}{dz}U(z,z_0)=i\big(H_0+V(z)\big)U(z,z_0),\quad U(z_0,z_0)=\mathbb{I}.
$$
The variation-of-constants formula gives, for $\mathbf{y}(0)=\mathbf{0}$,
$$
\mathbf{y}(L)=\int_0^L U(L,z)\,i\,\mathbf{b}(z)\,dz.
\tag{C}
$$

### Dyson series for the evolution operator

Because $V(z)$ need not commute at different $z$, pass to the interaction picture with respect to $H_0$:
$$
\mathbf{y}_I(z)=e^{-iH_0 z}\mathbf{y}(z),\qquad
V_I(z)=e^{-iH_0 z}V(z)e^{iH_0 z},\qquad
\mathbf{b}_I(z)=e^{-iH_0 z}\mathbf{b}(z).
$$
Then
$$
\frac{d\mathbf{y}_I}{dz}=iV_I(z)\mathbf{y}_I+i\mathbf{b}_I(z),\qquad \mathbf{y}_I(0)=\mathbf{0}.
\tag{D}
$$
The interaction-picture propagator is
$$
U_I(L,z)=\mathcal{T}_z\exp\left(i\int_z^L V_I(\zeta)\,d\zeta\right),
$$
so the **formal Dyson solution** is
$$
\begin{aligned}
\mathbf{y}_I(L) &= i\int_0^L U_I(L,z)\mathbf{b}_I(z)\,dz \\
&= i\int_0^L\left[\mathbb{I}
+i\int_z^L V_I(z_2)\,dz_2
+(i)^2\int_z^L dz_2 \int_{z_2}^L dz_3\,V_I(z_3)V_I(z_2)+\cdots\right]\mathbf{b}_I(z)\,dz.
\end{aligned}
\tag{E}
$$

We will only need terms up to $O(\kappa^2)$.

## 2) Second-order solution for $B_3(L)$ and equivalence to the NDPA formula

Compute the explicit interaction-picture objects:
$$
V_I(z)=\kappa(z)\begin{pmatrix}
0 & 2A_1^* e^{i\Delta k_2 z}\\[2pt]
3A_1 e^{-i\Delta k_2 z} & 0
\end{pmatrix},
\qquad
\mathbf{b}_I(z)=\kappa(z)\begin{pmatrix}A_1^2 e^{-i\Delta k_1 z}\\[2pt]0\end{pmatrix}.
\tag{F}
$$

*   **First order, $O(\kappa)$.**
    Keep only the identity inside the brackets in (E):
    $$
    \mathbf{y}_I^{(1)}(L)=i\int_0^L \mathbf{b}_I(z_1)\,dz_1
    = i\int_0^L \kappa(z_1)\begin{pmatrix}A_1^2 e^{-i\Delta k_1 z_1}\\[2pt]0\end{pmatrix}dz_1.
    $$
    This populates **only** the $B_2$ component; the $B_3$ component is still zero at this order.

*   **Second order, $O(\kappa^2)$.**
    Take the next term in (E), with one insertion of $V_I$:
    $$
    \mathbf{y}_I^{(2)}(L)
    = i^2\int_0^L dz_1\int_{z_1}^L dz_2\,V_I(z_2)\mathbf{b}_I(z_1).
    $$
    Using (F),
    $$
    V_I(z_2)\mathbf{b}_I(z_1)
    = \kappa(z_2)\kappa(z_1)
    \begin{pmatrix}
    0\\[2pt]
    3A_1^3 e^{-i(\Delta k_2 z_2+\Delta k_1 z_1)}
    \end{pmatrix}.
    $$
    Therefore the second component (the one that equals $e^{-i(\Delta k_1+\Delta k_2)L}B_3(L)$) is
    $$
    \begin{aligned}
    \big[\mathbf{y}_I^{(2)}(L)\big]_2
    &= i^2\cdot 3A_1^3\int_0^L dz_1\int_{z_1}^L dz_2\,
    \kappa(z_1)\kappa(z_2)e^{-i(\Delta k_1 z_1+\Delta k_2 z_2)} \\
    &= -3A_1^3 S[\kappa](\Delta k_1,\Delta k_2),
    \end{aligned}
    $$
    where
    $$
    S[\kappa](\Delta k_1,\Delta k_2)
    :=\int_0^L dz_2\int_0^{z_2} dz_1\,
    \kappa(z_1)\kappa(z_2) e^{-i(\Delta k_1 z_1+\Delta k_2 z_2)}
    $$
    is exactly the functional defined in **npda.md**.

Finally, relate pictures/frames. By definition of the interaction picture,
$$
\big[\mathbf{y}_I(L)\big]_2=e^{-i(\Delta k_1+\Delta k_2)L}B_3(L).
$$
The canonical transform between $\boldsymbol{A}$ and $\boldsymbol{B}$ is $B_3(z)=e^{i(\Delta k_1+\Delta k_2)z}A_3(z)$, hence at $z=L$
$$
A_3(L)=e^{-i(\Delta k_1+\Delta k_2)L}B_3(L)=\big[\mathbf{y}_I(L)\big]_2.
$$
Keeping terms up to $O(\kappa^2)$,
$$
A_3(L)= -3A_1^3 S[\kappa](\Delta k_1,\Delta k_2)
$$
with $A_1\equiv A_1(0)$. This is **exactly** the NDPA result from **npda.md**.

### Notes / comments

*   The overall factor **3** comes directly from the Manley–Rowe weight $J_{33}=3$ in the canonical equation, ensuring the Hamiltonian and coupled‑wave formalisms are consistent.
*   No additional terms contribute to $A_3$ at $O(\kappa^2)$: the inhomogeneous “source” $\mathbf{b}_I$ is $O(\kappa)$ and populates $B_2$, and a *single* interaction $V_I$ (also $O(\kappa)$) transfers amplitude to $B_3$. Any further insertions of $V_I$ would be $O(\kappa^3)$ or higher.
*   The derivation holds for a general (possibly sign‑alternating) piecewise-constant $\kappa(z)$, so it covers aperiodically poled QPM structures; the discretized evaluation in **npda.md** is just a fast way to compute $S[\kappa]$ numerically.