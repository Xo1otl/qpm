## 1) Put the model into a single well-defined integral

In the undepleted‑pump, slowly‑varying‑envelope approximation, the generated SH envelope can be written (up to the convention-dependent prefactors you’ve already fixed) as
[
A_2(L)= i,\kappa_{\rm mat},A_1^2 \int_0^L s(z),e^{i\Delta k z},dz,
]
where (s(z)\in{+1,-1}) is the sign pattern of the effective (\chi^{(2)}) (or (d)).

Your **shifted pulse per cell** corresponds to the following *exact* definition on each period (cell)
[
\text{cell }n:\quad z\in[n\Lambda_0,(n+1)\Lambda_0],\qquad z_n=n\Lambda_0+\frac{\Lambda_0}{2},
]
and introducing the local coordinate (z=z_n+\xi) with (\xi\in[-\Lambda_0/2,\Lambda_0/2]), define
[
s_n(\xi)=
\begin{cases}
-1,& \xi\in[\delta_n-w_n/2,\ \delta_n+w_n/2],[2pt]
+1,& \text{otherwise in }[-\Lambda_0/2,\Lambda_0/2].
\end{cases}
]
Equivalently:
[
s_n(\xi)=1-2,\mathbf{1}_{[\delta_n-w_n/2,\ \delta_n+w_n/2]}(\xi).
]

Then the total integral splits *exactly* into a discrete sum:
[
A_2(L)= i\kappa_{\rm mat}A_1^2 \sum_{n=0}^{N-1} \int_{-\Lambda_0/2}^{\Lambda_0/2} s_n(\xi),e^{i\Delta k(z_n+\xi)},d\xi
= i\kappa_{\rm mat}A_1^2 \sum_{n=0}^{N-1} e^{i\Delta k z_n},I_n(\Delta k),
]
where the “unit” integral (no approximation yet) is
[
I_n(\Delta k);:=;\int_{-\Lambda_0/2}^{\Lambda_0/2} s_n(\xi),e^{i\Delta k\xi},d\xi.
]

---

## 2) Exact evaluation of the unit-cell integral (I_n(\Delta k))

Using (s_n(\xi)=1-2\mathbf{1}*{\rm pulse}), you get the exact splitting
[
I_n(\Delta k)=\int*{-\Lambda_0/2}^{\Lambda_0/2} e^{i\Delta k\xi},d\xi
-2\int_{\delta_n-w_n/2}^{\delta_n+w_n/2} e^{i\Delta k\xi},d\xi.
]

### 2a) Background term (exact)

[
\int_{-\Lambda_0/2}^{\Lambda_0/2} e^{i\Delta k\xi},d\xi
=\left[\frac{e^{i\Delta k\xi}}{i\Delta k}\right]_{-\Lambda_0/2}^{\Lambda_0/2}
=\frac{2}{\Delta k},\sin!\Big(\frac{\Delta k\Lambda_0}{2}\Big).
]

### 2b) Pulse term (exact)

[
\int_{\delta_n-w_n/2}^{\delta_n+w_n/2} e^{i\Delta k\xi},d\xi
=\left[\frac{e^{i\Delta k\xi}}{i\Delta k}\right]_{\delta_n-w_n/2}^{\delta_n+w_n/2}
= e^{i\Delta k\delta_n},\frac{2}{\Delta k},\sin!\Big(\frac{\Delta k w_n}{2}\Big).
]

### 2c) Combine

So the exact closed form is:
[
I_n(\Delta k)=\frac{2}{\Delta k},\sin!\Big(\frac{\Delta k\Lambda_0}{2}\Big)
-\frac{4}{\Delta k},e^{i\Delta k\delta_n},\sin!\Big(\frac{\Delta k w_n}{2}\Big).
]

This is the rigorous starting point.

---

## 3) The phase-aliasing factor (({-1})^m) is exact (given the standard QPM definition)

Define the (m)-th grating vector
[
G_m=\frac{2\pi m}{\Lambda_0},
]
and define the residual mismatch in the usual QPM way:
[
\Delta\beta := \Delta k - G_m
\quad\Longleftrightarrow\quad
\Delta k = G_m + \Delta\beta.
]

Then
[
e^{i\Delta k z_n} = e^{i(G_m+\Delta\beta)z_n}=e^{i\Delta\beta z_n},e^{iG_m z_n}.
]
But (z_n=(n+\tfrac12)\Lambda_0), so
[
e^{iG_m z_n} = \exp!\left(i\frac{2\pi m}{\Lambda_0}(n+\tfrac12)\Lambda_0\right)
=\exp!\left(i2\pi mn\right)\exp(i\pi m)
= (-1)^m.
]
Therefore the “aliasing” relation
[
e^{i\Delta k z_n}=(-1)^m e^{i\Delta\beta z_n}
]
is **exact**, not approximate, once (\Delta\beta=\Delta k-G_m) and (z_n=n\Lambda_0+\Lambda_0/2) are adopted.

---

## 4) Your (I_{\rm unit}(n)) matches the exact (I_n(\Delta k)) at (\Delta k=G_m)

You define
[
I_{\rm unit}(n) = \int_{-\Lambda_0/2}^{\Lambda_0/2} (+1)e^{i G_m \xi},d\xi
-2\int_{\delta_n-w_n/2}^{\delta_n+w_n/2} (+1)e^{iG_m \xi},d\xi.
]
This is exactly (I_n(\Delta k)) evaluated at (\Delta k=G_m), i.e. (I_{\rm unit}(n)=I_n(G_m)).

Plugging (\Delta k = G_m) into the exact formula from §2:

### 4a) Background vanishes for (m\neq 0)

[
\int_{-\Lambda_0/2}^{\Lambda_0/2} e^{iG_m\xi},d\xi=\frac{2}{G_m}\sin!\Big(\frac{G_m\Lambda_0}{2}\Big)
=\frac{2}{G_m}\sin(\pi m)=0,\qquad m\in\mathbb Z\setminus{0}.
]
So the “background vanishes” statement is rigorous **for the choice (G_m=2\pi m/\Lambda_0)** and **(m\neq 0)**.

### 4b) Pulse term gives your expression

Using the pulse integral above with (\Delta k=G_m):
[
\int_{\delta_n-w_n/2}^{\delta_n+w_n/2} e^{iG_m\xi},d\xi
= e^{iG_m\delta_n},\frac{2}{G_m}\sin!\Big(\frac{G_m w_n}{2}\Big).
]
Therefore
[
I_{\rm unit}(n)= -2 \times e^{iG_m\delta_n},\frac{2}{G_m}\sin!\Big(\frac{G_m w_n}{2}\Big)
= -\frac{4}{G_m},e^{iG_m\delta_n},\sin!\Big(\frac{G_m w_n}{2}\Big).
]

Now substitute (G_m=2\pi m/\Lambda_0), (D_n=w_n/\Lambda_0), and (\phi_n:=G_m\delta_n). Then
[
\sin!\Big(\frac{G_m w_n}{2}\Big)=\sin!\Big(\frac{2\pi m}{\Lambda_0}\frac{w_n}{2}\Big)
=\sin(\pi m D_n),
]
and
[
\frac{4}{G_m}=\frac{4}{2\pi m/\Lambda_0}=\frac{2\Lambda_0}{\pi m}.
]
So
[
I_{\rm unit}(n)= -\Lambda_0,\frac{2}{\pi m},\sin(\pi m D_n),e^{i\phi_n},
]
which is exactly your
[
I_{\text{unit}}(n) = - \Lambda_0 \frac{2}{m \pi} \sin(m \pi D_n) e^{i \phi_n}.
]

✅ **Unit-cell evaluation is correct.**

---

## 5) The only approximation: replacing (\Delta k) by (G_m) *inside* the unit-cell integral

Your discrete expression uses (I_{\rm unit}(n)) computed with (G_m), while the exact cell integral would use (\Delta k):
[
I_n(\Delta k) \quad\text{vs}\quad I_n(G_m).
]

This is the “QPM (\Delta k\approx G_m)” step. To verify rigorously that it is harmless in the claimed continuum limit, quantify the error.

Write (\Delta k = G_m+\Delta\beta). Then from the exact background term:
[
\frac{2}{\Delta k}\sin!\Big(\frac{\Delta k\Lambda_0}{2}\Big)
= \frac{2}{G_m+\Delta\beta}\sin(\pi m + \tfrac{\Delta\beta\Lambda_0}{2})
= \frac{2}{G_m+\Delta\beta}(-1)^m\sin!\Big(\frac{\Delta\beta\Lambda_0}{2}\Big).
]
For small (|\Delta\beta|\Lambda_0),
[
\sin!\Big(\frac{\Delta\beta\Lambda_0}{2}\Big)=\mathcal O(\Delta\beta\Lambda_0),
\qquad
\frac{1}{G_m+\Delta\beta}=\mathcal O(\Lambda_0),
]
since (G_m\sim 1/\Lambda_0). Hence the background term scales like
[
\text{background}=\mathcal O(\Delta\beta,\Lambda_0^2).
]

A similar Taylor expansion shows the pulse term difference
[
I_n(\Delta k)-I_n(G_m)=\mathcal O(\Delta\beta,\Lambda_0^2)
]
provided (D_n) stays bounded in ([0,1]) and (\delta_n=\mathcal O(\Lambda_0)) (which it is under your scaling (\delta_n=\tfrac{\phi_n}{2\pi m}\Lambda_0) with bounded (\phi_n)).

Now sum over (N=L/\Lambda_0) cells. The *total* approximation error becomes
[
\sum_{n=0}^{N-1} \mathcal O(\Delta\beta,\Lambda_0^2)
= \mathcal O(\Delta\beta,N\Lambda_0^2)
= \mathcal O(\Delta\beta,L,\Lambda_0)\xrightarrow[\Lambda_0\to 0]{}0.
]

✅ So in the stated continuum limit (\Lambda_0\to 0) with (L) fixed, **your replacement (\Delta k\mapsto G_m) inside (I_{\rm unit})** is rigorously justified: its cumulative contribution vanishes.

(For a finite physical (\Lambda_0), the same estimate says the approximation is accurate when (|\Delta\beta|\Lambda_0\ll 1).)

---

## 6) Continuum limit as a midpoint Riemann sum

After aliasing and inserting (I_{\rm unit}), your discrete sum is
[
A_2(L)\approx i\kappa_{\rm mat}A_1^2
\sum_{n=0}^{N-1}
(-1)^m e^{i\Delta\beta z_n}
\left[-\Lambda_0\frac{2}{m\pi}\sin(m\pi D_n)e^{i\phi_n}\right].
]
Combine the signs:
[
A_2(L)\approx i\kappa_{\rm mat}A_1^2
\sum_{n=0}^{N-1}
\left[(-1)^{m+1}\frac{2}{m\pi}\sin(m\pi D_n)e^{i\phi_n}\right]
e^{i\Delta\beta z_n},\Lambda_0.
]

Define the sampled function
[
f(z_n):=\left[(-1)^{m+1}\frac{2}{m\pi}\sin(m\pi D(z_n))e^{i\phi(z_n)}\right]e^{i\Delta\beta z_n}.
]
Then the sum is precisely the **midpoint rule**
[
\sum_{n=0}^{N-1} f(z_n),\Lambda_0.
]

A standard theorem from real analysis says:

* If (f) is **Riemann integrable** on ([0,L]) (e.g., if (D(z)) and (\phi(z)) are bounded and piecewise continuous, which makes (f) bounded and piecewise continuous), then as the mesh size (\max_n |I_n|=\Lambda_0\to 0),
  [
  \sum_{n=0}^{N-1} f(z_n),\Lambda_0 ;\longrightarrow; \int_0^L f(z),dz.
  ]

Therefore,
[
A_2(L)\to i\kappa_{\rm mat}A_1^2
\int_0^L
\left[(-1)^{m+1}\frac{2}{m\pi}\sin(m\pi D(z))e^{i\phi(z)}\right]
e^{i\Delta\beta z},dz.
]

✅ This exactly matches your continuum expression.

---

## 7) Complex effective nonlinearity

From the last integral, it is natural (and correct) to define the **local complex effective coupling**
[
\boxed{;
\kappa_{\rm eff}(z)=(-1)^{m+1},\kappa_{\rm mat},\frac{2}{m\pi},\sin(m\pi D(z)),e^{i\phi(z)}; }.
]

* The **magnitude** is controlled by (\sin(m\pi D)) (duty cycle).
* The **phase** is controlled by (\phi(z)=G_m\delta(z)) (shift).

---

## 8) Technical conditions and edge cases (what you must assume for “rigor”)

To make every step fully rigorous, you need:

1. **Cell containment (no wrap‑around)** for the pulse interval inside each unit cell, or else you must treat wrap-around by splitting the pulse integral at the cell boundary. A sufficient condition is
   [
   |\delta_n|\le \frac{\Lambda_0-w_n}{2}.
   ]
   Under your scaling (\delta_n=\mathcal O(\Lambda_0)) with bounded (\phi_n), this is typically satisfiable when (D_n<1).

2. **Integrability / regularity** for the continuum limit:
   It suffices that (D(z)) and (\phi(z)) are bounded and piecewise continuous on ([0,L]). Then
   (\sin(m\pi D(z))e^{i\phi(z)}e^{i\Delta\beta z}) is Riemann integrable, guaranteeing convergence of the midpoint sums.

3. **Order (m\neq 0)** for the “background vanishes” statement.
   For (m=0), (G_0=0), the background integral is (\Lambda_0) and you must treat that case separately (it’s not QPM).

4. **Near-QPM approximation inside the cell**:
   The replacement (I_n(\Delta k)\to I_n(G_m)) is controlled by (|\Delta\beta|\Lambda_0). In the strict continuum limit (\Lambda_0\to 0), the total error vanishes as (\mathcal O(\Delta\beta L\Lambda_0)).

---

## Bottom line

* Your **unit-cell integral** (I_{\rm unit}(n)) is exactly correct for the shifted rectangular pulse with background (+1) and inverted pulse (-1).
* Your **aliasing factor** (e^{i\Delta k z_n}\mapsto (-1)^m e^{i\Delta\beta z_n}) is exact given (\Delta\beta=\Delta k-G_m) and (z_n=(n+\tfrac12)\Lambda_0).
* Your **continuum limit** is a standard midpoint Riemann‑sum limit, valid whenever (D(z)) and (\phi(z)) are bounded and (piecewise) continuous / Riemann integrable.
* The only “hidden” approximation—using (G_m) instead of (\Delta k) inside the unit-cell integral—is rigorously harmless in the (\Lambda_0\to 0) limit and is small when (|\Delta\beta|\Lambda_0\ll 1) at finite (\Lambda_0).

If you want, I can also rewrite the entire derivation in a “single-line” homogenization/Fourier‑coefficient form (showing (\kappa_{\rm eff}(z)) is precisely the local (m)-th spatial Fourier coefficient of the sign function (s(z)) in complex form), which makes the origin of the phase factor (e^{i\phi(z)}) completely transparent.
