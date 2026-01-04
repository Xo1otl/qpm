# Coupled-Wave Equations (CWE) for SHG
SHG Coupled-Wave Equations (CWE):

$$\frac{d}{dz} A_\omega(z) = -j\kappa(z)^*[A_\omega(z)]^*A_{2\omega}(z)e^{-j(\Delta k)z}$$

$$\frac{d}{dz} A_{2\omega}(z) = -j\kappa(z)[A_\omega(z)]^2e^{+j(\Delta k)z}$$

$A_\omega, A_{2\omega}$: complex amplitudes.

$\kappa(z)$: nonlinear coupling coefficient.

$\Delta k = \beta^{2\omega} - 2\beta^{\omega}$: phase mismatch.

# Non-Pump-Depletion Approximation (NPDA)

NPDA:

$$A_\omega(z) \approx A_{\omega}(z^{(0)}) \quad \text{(constant)}$$

CWE simplifies:

$$\frac{d}{dz} A_{2\omega}(z) = -j\kappa(z)A_{\omega}(z^{(0)})^2e^{+j(\Delta k)z}$$

# SHG in Quasi-Phase-Matching (QPM) Structures

Integrate simplified CWE from $z^{(0)}$ to $z^{(N)}$:

$$A_{2\omega}(z^{(N)}) - A_{2\omega}(z^{(0)}) = -jA_{\omega}(z^{(0)})^2 \int_{z^{(0)}}^{z^{(N)}} \kappa(z)e^{+j\Delta k z} dz$$

Assume $A_{2\omega}(z^{(0)}) = 0$:

$$A_{2\omega}(z^{(N)}) = -jA_{\omega}(z^{(0)})^2 \int_{z^{(0)}}^{z^{(N)}} \kappa(z)e^{+j\Delta k z} dz$$

QPM: $\kappa(z)$ is piecewise constant. Decompose integral over N layers:

$$A_{2\omega}(z^{(N)}) = -jA_{\omega}(z^{(0)})^2 \sum_{k=0}^{N-1} \kappa^{(k)} \int_{z^{(k)}}^{z^{(k+1)}} e^{+j\Delta k z} dz$$

Layer integral solution:

$$\int_{z^{(k)}}^{z^{(k+1)}} e^{+j\Delta k z} dz = L^{(k)} e^{j\frac{\Delta k}{2} L^{(k)}} \cdot \frac{\sin(\frac{\Delta k}{2} L^{(k)})}{\frac{\Delta k}{2} L^{(k)}} \cdot e^{j\Delta k z^{(k)}}$$

$L^{(k)} = z^{(k+1)} - z^{(k)}$: k-th layer length.

Total output (coherent sum):

$$A_{2\omega}(z^{(N)}) = \sum_{k=0}^{N-1} \tilde{A}_{2\omega}^{(k)} \cdot e^{j\Delta k z^{(k)}}$$

k-th layer contribution:

$$\tilde{A}_{2\omega}^{(k)} = -j\kappa^{(k)}A_{\omega}(z^{(0)})^2 L^{(k)} e^{j\frac{\Delta k}{2} L^{(k)}} \cdot \frac{\sin(\frac{\Delta k}{2} L^{(k)})}{\frac{\Delta k}{2} L^{(k)}}$$

# Special Case: Uniformly Poled Structure

Conditions:

$L^{(k)} = \frac{\Lambda}{2}$ (constant layer thickness)

$\kappa^{(k)} = \kappa_{mag}(-1)^k$ (periodic inversion)

Effective phase mismatch:

$$\Delta k' = \beta^{2\omega} - 2\beta^{\omega} - K = \Delta k - K$$

$K = \frac{2\pi}{\Lambda}$ (grating vector).

### Step 1: Apply Conditions

$$\tilde{A}_{2\omega}^{(k)} = -j\kappa_{mag}(-1)^k \underbrace{A_{\omega}(z^{(0)})^2 \frac{\Lambda}{2} e^{j\frac{\Delta k}{2} \frac{\Lambda}{2}} \frac{\sin(\frac{\Delta k}{2} \frac{\Lambda}{2})}{\frac{\Delta k}{2} \frac{\Lambda}{2}}}_{C}$$

C is constant. $z^{(k)} = k \frac{\Lambda}{2}$.

$$A_{2\omega}(z^{(N)}) = \sum_{k=0}^{N-1} \left( -j\kappa_{mag}(-1)^k C \right) \cdot e^{j\Delta k (k\Lambda/2)}$$

$$A_{2\omega}(z^{(N)}) = -j\kappa_{mag} C \sum_{k=0}^{N-1} (-1)^k e^{j\frac{\Delta k}{2} k \Lambda}$$

### Step 2: Solve Geometric Series

$(-1)^k = e^{jk\pi}$

$$A_{2\omega}(z^{(N)}) = -j\kappa_{mag} C \sum_{k=0}^{N-1} e^{jk\pi} e^{j\frac{\Delta k}{2} k \Lambda} = -j\kappa_{mag} C \sum_{k=0}^{N-1} (e^{j(\pi + \frac{\Delta k}{2} \Lambda)})^k$$

$\alpha = \pi + \frac{\Delta k}{2}\Lambda$.

$$\sum_{k=0}^{N-1} e^{jk\alpha} = \frac{1-e^{jN\alpha}}{1-e^{j\alpha}} = \frac{\sin(N\alpha/2)}{\sin(\alpha/2)}e^{j(N-1)\alpha/2}$$

### Step 3: Introduce $\frac{\Delta k'}{2}$

$\frac{\Delta k}{2}\Lambda = \frac{\Delta k'}{2}\Lambda + \pi$.

$$\alpha = \pi + (\frac{\Delta k'}{2}\Lambda + \pi) = 2\pi + \frac{\Delta k'}{2}\Lambda$$

Summation term:

$$\sum = \frac{\sin(N(\pi + \frac{\Delta k'}{2}\Lambda/2))}{\sin(\pi + \frac{\Delta k'}{2}\Lambda/2)}e^{j(N-1)(\pi + \frac{\Delta k'}{2}\Lambda/2)}$$

Step 4: Simplify Sum (Near-phase-matched $\frac{\Delta k'}{2} \approx 0$, $N \gg 1$)

Ratio Term: $\sin(x+n\pi) = (-1)^n\sin(x)$, $\sin(x) \approx x$ (denominator).

$$\frac{\sin(N\pi + N\frac{\Delta k'}{2}\Lambda/2)}{\sin(\pi + \frac{\Delta k'}{2}\Lambda/2)} = (-1)^{N+1} \frac{\sin(N\frac{\Delta k'}{2}\Lambda/2)}{\sin(\frac{\Delta k'}{2}\Lambda/2)} \approx (-1)^{N+1} \frac{\sin(N\frac{\Delta k'}{2}\Lambda/2)}{\frac{\Delta k'}{2}\Lambda/2}$$

$z^{(N)} = N \Lambda/2$.

$$\text{Ratio} \approx (-1)^{N+1} N \frac{\sin(\frac{\Delta k'}{2}z^{(N)})}{\frac{\Delta k'}{2}z^{(N)}}$$

Phase Term:

$$e^{j(N-1)(\pi + \frac{\Delta k'}{2}\Lambda/2)} = (-1)^{N-1} e^{j(N-1)\frac{\Delta k'}{2}\Lambda/2} \approx (-1)^{N-1} e^{j\frac{\Delta k'}{2}z^{(N)}}$$

### Step 5: Simplify C ($\frac{\Delta k'}{2} \approx 0$)

$\frac{\Delta k}{2}\Lambda/2 = \frac{\Delta k'}{2}\Lambda/2 + \pi/2$.

$$C = A_{\omega}(z^{(0)})^2 \frac{\Lambda}{2} e^{j\frac{\Delta k}{2} \frac{\Lambda}{2}} \frac{\sin(\frac{\Delta k}{2} \frac{\Lambda}{2})}{\frac{\Delta k}{2} \frac{\Lambda}{2}}$$

$e^{j\frac{\Delta k}{2} \frac{\Lambda}{2}} = e^{j(\frac{\Delta k'}{2}\Lambda/2 + \pi/2)} = j e^{j\frac{\Delta k'}{2}\Lambda/2}$.

$\frac{\sin(\frac{\Delta k}{2} \frac{\Lambda}{2})}{\frac{\Delta k}{2} \frac{\Lambda}{2}} = \frac{\sin(\frac{\Delta k'}{2}\Lambda/2 + \pi/2)}{\frac{\Delta k'}{2}\Lambda/2 + \pi/2} = \frac{\cos(\frac{\Delta k'}{2}\Lambda/2)}{\frac{\Delta k'}{2}\Lambda/2 + \pi/2} \approx \frac{1}{\pi/2} = \frac{2}{\pi}$.

$$C \approx A_{\omega}(z^{(0)})^2 \frac{\Lambda}{2} \cdot \left(j e^{j\frac{\Delta k'}{2}\Lambda/2}\right) \cdot \left(\frac{2}{\pi}\right) = j \frac{A_{\omega}(z^{(0)})^2 \Lambda}{\pi} e^{j\frac{\Delta k'}{2}\Lambda/2}$$

### Step 6: Combine Terms

$$A_{2\omega}(z^{(N)}) \approx -j\kappa_{mag} \cdot \left(j \frac{A_{\omega}^2 \Lambda}{\pi} e^{j\frac{\Delta k'}{2}\Lambda/2}\right) \cdot \left((-1)^{N+1} N \frac{\sin(\frac{\Delta k'}{2}z^{(N)})}{\frac{\Delta k'}{2}z^{(N)}}\right) \cdot \left((-1)^{N-1} e^{j\frac{\Delta k'}{2}z^{(N)}}\right)$$

$-j \cdot j = 1$. $(-1)^{N+1} \cdot (-1)^{N-1} = 1$.

$$A_{2\omega}(z^{(N)}) \approx \kappa_{mag} \frac{A_{\omega}^2 \Lambda N}{\pi} e^{j\frac{\Delta k'}{2}\Lambda/2} \frac{\sin(\frac{\Delta k'}{2}z^{(N)})}{\frac{\Delta k'}{2}z^{(N)}} e^{j\frac{\Delta k'}{2}z^{(N)}}$$

$\Lambda N = 2z^{(N)}$. $e^{j\frac{\Delta k'}{2}\Lambda/2} = e^{j\frac{\Delta k'}{2}z^{(N)}/N} \approx 1$ (for $N \gg 1$).

$$A_{2\omega}(z^{(N)}) \approx \kappa_{mag} \frac{A_{\omega}^2 (2z^{(N)})}{\pi} \frac{\sin(\frac{\Delta k'}{2}z^{(N)})}{\frac{\Delta k'}{2}z^{(N)}} e^{j\frac{\Delta k'}{2}z^{(N)}}$$

### Final Result

$$A_{2\omega}(z^{(N)}) \approx \frac{2}{\pi}\kappa_{mag} A_{\omega}(z^{(0)})^2 z^{(N)} e^{j\frac{\Delta k'}{2} z^{(N)}} \frac{\sin(\frac{\Delta k'}{2} z^{(N)})}{\frac{\Delta k'}{2} z^{(N)}}$$

Effective nonlinear coefficient reduced by $2/\pi$. Output follows $\text{sinc}(\frac{\Delta k'}{2} z^{(N)})$.

# Fourier Transform Relationship

Define device function $\kappa_{device}(z)$ ($\kappa(z)$ for $z^{(0)} \le z \le z^{(N)}$, 0 elsewhere).

$$A_{2\omega}(z^{(N)}) = -jA_{\omega}(z^{(0)})^2 \int_{-\infty}^{\infty} \kappa_{device}(z)e^{+j\Delta k z} dz$$

Fourier transform definition:

$$\mathcal{F}[f(z)](k) = \tilde{f}(k) = \int_{-\infty}^{\infty} f(z) e^{-ikz} dz$$

SHG integral is Fourier transform of $\kappa_{device}(z)$ at $k = -\Delta k$:

$$\int_{-\infty}^{\infty} \kappa_{device}(z)e^{+j\Delta k z} dz = \tilde{\kappa}_{device}(-\Delta k)$$

Output amplitude:

$$A_{2\omega}(z^{(N)}) = -jA_{\omega}(z^{(0)})^2 \cdot \tilde{\kappa}_{device}(-\Delta k)$$

QPM design $\equiv$ designing $\kappa(z)$ for target Fourier transform $\tilde{\kappa}_{device}(-\Delta k)$.
