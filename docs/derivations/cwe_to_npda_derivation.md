# Introduction

The coupled-wave equations (CWE) that describe second-harmonic generation (SHG) are fundamental to understanding nonlinear optical interactions. However, their complexity can make them difficult to solve analytically. For many practical applications, such as in high-power laser systems, the power of the fundamental wave does not significantly deplete during the interaction.

This allows for the use of the non-pump-depletion approximation (NPDA), which simplifies the CWE. This document provides a step-by-step derivation of this simplified model and explores its relationship with quasi-phase-matching structures and Fourier analysis.

# Overview

This document details the derivation of the second-harmonic wave amplitude under the **Non-Pump-Depletion Approximation (NPDA)**. We begin with the general **coupled-wave equations (CWE)** for SHG and apply the NPDA to simplify them. We then integrate the simplified equation over the device length for a generic **quasi-phase-matching (QPM)** structure. We then derive the specific solution for a **uniformly poled structure** with a constant period, showing how the general solution simplifies to a practical formula. Finally, we show that the resulting expression is equivalent to the **Fourier transform** of the nonlinear coefficient profile, providing a powerful tool for device design.

# Coupled-Wave Equations (CWE) for SHG

In an SHG device, the interaction between the fundamental wave (at frequency ω) and the second-harmonic wave (at frequency 2ω) is described by the following nonlinear coupled-mode equations:

$$
\frac{d}{dz} A_\omega(z) = -j\boldsymbol\kappa(z)^*[A_\omega(z)]^*A_{2\omega}(z)e^{-j(2\Gamma)z}
$$
$$ 
\frac{d}{dz} A_{2\omega}(z) = -j\boldsymbol\kappa(z)[A_\omega(z)]^2e^{+j(2\Gamma)z}
$$ 

Here, $A_\omega(z)$ and $A_{2\omega}(z)$ are the complex amplitudes of the fundamental and second-harmonic waves, respectively, $\kappa(z)$ is the nonlinear coupling coefficient, and $2\Gamma = \beta^{2\omega} - 2\beta^{\omega}$ represents the phase mismatch between the waves.

# Non-Pump-Depletion Approximation (NPDA)

Under the NPDA, we assume that the amplitude of the fundamental wave remains nearly constant throughout the propagation:
$$A_\omega(z) \approx A_{\omega}(z^{(0)}) \quad \text{(constant)}$$

This approximation allows us to disregard the first CWE and simplifies the second equation to:
$$\frac{d}{dz} A_{2\omega}(z) = -j\kappa(z)A_{\omega}(z^{(0)})^2e^{+j(2\Gamma)z}$$

# SHG in Quasi-Phase-Matching (QPM) Structures

To find the output amplitude of the second-harmonic wave, we integrate the simplified differential equation over the entire length of the device, from $z^{(0)}$ to $z^{(N)}$.

$$A_{2\omega}(z^{(N)}) - A_{2\omega}(z^{(0)}) = -jA_{\omega}(z^{(0)})^2 \int_{z^{(0)}}^{z^{(N)}} \kappa(z)e^{+j2\Gamma z} dz$$

Assuming no initial second-harmonic wave, $A_{2\omega}(z^{(0)}) = 0$, the equation becomes:

$$A_{2\omega}(z^{(N)}) = -jA_{\omega}(z^{(0)})^2 \int_{z^{(0)}}^{z^{(N)}} \kappa(z)e^{+j2\Gamma z} dz$$

In a QPM structure, the nonlinear coefficient $\kappa(z)$ is constant within each domain (layer) but varies between them. We can thus decompose the integral into a sum over N layers:

$$A_{2\omega}(z^{(N)}) = -jA_{\omega}(z^{(0)})^2 \sum_{k=0}^{N-1} \kappa^{(k)} \int_{z^{(k)}}^{z^{(k+1)}} e^{+j2\Gamma z} dz$$

Solving the integral for each layer gives:

$$ 
\int_{z^{(k)}}^{z^{(k+1)}} e^{+j2\Gamma z} dz = L^{(k)} e^{j\Gamma L^{(k)}} \cdot \frac{\sin(\Gamma L^{(k)})}{\Gamma L^{(k)}} \cdot e^{j2\Gamma z^{(k)}}
$$ 

where $L^{(k)} = z^{(k+1)} - z^{(k)}$ is the length of the k-th layer.

The total output amplitude is the coherent sum of the contributions from all layers:

$$ 
A_{2\omega}(z^{(N)}) = \sum_{k=0}^{N-1} \tilde{A}_{2\omega}^{(k)} \cdot e^{j2\Gamma z^{(k)}}
$$ 

where the contribution from the k-th layer is:

$$ 
\tilde{A}_{2\omega}^{(k)} = -j\kappa^{(k)}A_{\omega}(z^{(0)})^2 L^{(k)} e^{j\Gamma L^{(k)}} \cdot \frac{\sin(\Gamma L^{(k)})}{\Gamma L^{(k)}}
$$ 

# Special Case: Uniformly Poled Structure

Let's consider a QPM device with a constant poling period, under the following conditions:
1.  **Constant layer thickness**: $L^{(k)} = \frac{\Lambda}{2}$ for all layers.
2.  **Periodic inversion of the nonlinear coefficient**: $\kappa^{(k)} = \kappa_{mag}(-1)^k$.

We define an effective phase mismatch $2\Gamma'$ that accounts for the QPM grating period:
$$2\Gamma' = \beta^{2\omega} - 2\beta^{\omega} - K = 2\Gamma - K$$
where $K = \frac{2\pi}{\Lambda}$ is the grating vector.

Our goal is to derive the simplified expression for the output amplitude $A_{2\omega}(z^{(N)})$ under these conditions.

### Step 1: Apply Conditions to the General Formula

Substituting the conditions into the expression for $\tilde{A}_{2\omega}^{(k)}$:
$$
\tilde{A}_{2\omega}^{(k)} = -j\kappa_{mag}(-1)^k \underbrace{A_{\omega}(z^{(0)})^2 \frac{\Lambda}{2} e^{j\Gamma \frac{\Lambda}{2}} \frac{\sin(\Gamma \frac{\Lambda}{2})}{\Gamma \frac{\Lambda}{2}}}_{C}
$$ 

The term C is a constant for all layers. With the layer position $z^{(k)} = k \frac{\Lambda}{2}$, the total amplitude becomes:
$$A_{2\omega}(z^{(N)}) = \sum_{k=0}^{N-1} \left( -j\kappa_{mag}(-1)^k C \right) \cdot e^{j2\Gamma (k\Lambda/2)}$$
$$A_{2\omega}(z^{(N)}) = -j\kappa_{mag} C \sum_{k=0}^{N-1} (-1)^k e^{j\Gamma k \Lambda}$$

### Step 2: Solve the Geometric Series

Using the identity $(-1)^k = e^{jk\pi}$, we can write the sum as a geometric series:
$$A_{2\omega}(z^{(N)}) = -j\kappa_{mag} C \sum_{k=0}^{N-1} e^{jk\pi} e^{j\Gamma k \Lambda} = -j\kappa_{mag} C \sum_{k=0}^{N-1} (e^{j(\pi + \Gamma \Lambda)})^k$$
With $\alpha = \pi + \Gamma\Lambda$, the sum of the series is:
$$\sum_{k=0}^{N-1} e^{jk\alpha} = \frac{1-e^{jN\alpha}}{1-e^{j\alpha}} = \frac{\sin(N\alpha/2)}{\sin(\alpha/2)}e^{j(N-1)\alpha/2}$$

### Step 3: Introduce Effective Phase Mismatch

We relate $\alpha$ to $\Gamma'$ using $\Gamma\Lambda = \Gamma'\Lambda + \pi$.
$$\alpha = \pi + (\Gamma'\Lambda + \pi) = 2\pi + \Gamma'\Lambda$$

The summation term becomes:
$$\sum = \frac{\sin(N(\pi + \Gamma'\Lambda/2))}{\sin(\pi + \Gamma'\Lambda/2)}e^{j(N-1)(\pi + \Gamma'\Lambda/2)}$$

### Step 4: Simplify the Summation Term

We simplify the ratio and phase terms separately, assuming a near-phase-matched condition ($\Gamma' \approx 0$) and a large number of layers ($N \gg 1$).

- **Ratio Term**: Using $\sin(x+n\pi) = (-1)^n\sin(x)$ and the small-angle approximation $\sin(x) \approx x$ for the denominator:
$$\frac{\sin(N\pi + N\Gamma'\Lambda/2)}{\sin(\pi + \Gamma'\Lambda/2)} = (-1)^{N+1} \frac{\sin(N\Gamma'\Lambda/2)}{\sin(\Gamma'\Lambda/2)} \approx (-1)^{N+1} \frac{\sin(N\Gamma'\Lambda/2)}{\Gamma'\Lambda/2}$$
With the total device length $z^{(N)} = N \Lambda/2$, this simplifies to:
$$\text{Ratio} \approx (-1)^{N+1} N \frac{\sin(\Gamma'z^{(N)})}{\Gamma'z^{(N)}}$$

- **Phase Term**:
$$e^{j(N-1)(\pi + \Gamma'\Lambda/2)} = (-1)^{N-1} e^{j(N-1)\Gamma'\Lambda/2} \approx (-1)^{N-1} e^{j\Gamma'z^{(N)}}$$

### Step 5: Simplify the Constant C

We substitute $\Gamma\Lambda/2 = \Gamma'\Lambda/2 + \pi/2$ into the expression for C.
$$C = A_{\omega}(z^{(0)})^2 \frac{\Lambda}{2} e^{j\Gamma \frac{\Lambda}{2}} \frac{\sin(\Gamma \frac{\Lambda}{2})}{\Gamma \frac{\Lambda}{2}}$$
The exponential term becomes $e^{j(\Gamma'\Lambda/2 + \pi/2)} = j e^{j\Gamma'\Lambda/2}$.
The sinc term becomes $\frac{\sin(\Gamma'\Lambda/2 + \pi/2)}{\Gamma'\Lambda/2 + \pi/2} = \frac{\cos(\Gamma'\Lambda/2)}{\Gamma'\Lambda/2 + \pi/2}$.
For $\Gamma' \approx 0$, this sinc term approximates to $\frac{1}{\pi/2} = \frac{2}{\pi}$.
$$C \approx A_{\omega}(z^{(0)})^2 \frac{\Lambda}{2} \cdot \left(j e^{j\Gamma'\Lambda/2}\right) \cdot \left(\frac{2}{\pi}\right) = j \frac{A_{\omega}(z^{(0)})^2 \Lambda}{\pi} e^{j\Gamma'\Lambda/2}$$

### Step 6: Combine All Terms

Assembling the full expression:
$$A_{2\omega}(z^{(N)}) \approx -j\kappa_{mag} \cdot \left(j \frac{A_{\omega}^2 \Lambda}{\pi} e^{j\Gamma'\Lambda/2}\right) \cdot \left((-1)^{N+1} N \frac{\sin(\Gamma'z^{(N)})}{\Gamma'z^{(N)}}\right) \cdot \left((-1)^{N-1} e^{j\Gamma'z^{(N)}}\right)$$
The prefactors simplify: $-j \cdot j = 1$ and $(-1)^{N+1} \cdot (-1)^{N-1} = 1$.
$$A_{2\omega}(z^{(N)}) \approx \kappa_{mag} \frac{A_{\omega}^2 \Lambda N}{\pi} e^{j\Gamma'\Lambda/2} \frac{\sin(\Gamma'z^{(N)})}{\Gamma'z^{(N)}} e^{j\Gamma'z^{(N)}}$$
Using $\Lambda N = 2z^{(N)}$ and approximating $e^{j\Gamma'\Lambda/2} = e^{j\Gamma'z^{(N)}/N} \approx 1$ for $N \gg 1$:
$$A_{2\omega}(z^{(N)}) \approx \kappa_{mag} \frac{A_{\omega}^2 (2z^{(N)})}{\pi} \frac{\sin(\Gamma'z^{(N)})}{\Gamma'z^{(N)}} e^{j\Gamma'z^{(N)}}$$

### Final Result

The final expression for the second-harmonic amplitude in a uniformly poled QPM structure is:
$$A_{2\omega}(z^{(N)}) \approx \frac{2}{\pi}\kappa_{mag} A_{\omega}(z^{(0)})^2 z^{(N)} e^{j\Gamma' z^{(N)}} \frac{\sin(\Gamma' z^{(N)})}{\Gamma' z^{(N)}}$$

This result shows that for a periodic QPM structure, the effective nonlinear coefficient is reduced by a factor of $2/\pi$ compared to an ideal phase-matched material. The output amplitude follows a sinc function with respect to the effective phase mismatch $\Gamma'$, which is characteristic of phase-matching processes.

# Fourier Transform Relationship

The expression for the SHG output can be elegantly connected to the Fourier transform. By defining a device function $\kappa_{device}(z)$ that is equal to $\kappa(z)$ within the device ($z^{(0)} \le z \le z^{(N)}$) and zero elsewhere, we can extend the integration limits to infinity without changing the result:

$$ 
A_{2\omega}(z^{(N)}) = -jA_{\omega}(z^{(0)})^2 \int_{-\infty}^{\infty} \kappa_{device}(z)e^{+j2\Gamma z} dz
$$ 

Recalling the definition of the Fourier transform:

$$ 
\mathcal{F}[f(z)](k) = \tilde{f}(k) = \int_{-\infty}^{\infty} f(z) e^{-ikz} dz
$$ 

We can see that the integral for the SHG output is the Fourier transform of $\kappa_{device}(z)$ evaluated at $k = -2\Gamma$:

$$ 
\int_{-\infty}^{\infty} \kappa_{device}(z)e^{+j2\Gamma z} dz = \tilde{\kappa}_{device}(-2\Gamma)
$$ 

Thus, the output amplitude is directly proportional to the Fourier transform of the nonlinear coefficient profile:

$$ A_{2\omega}(z^{(N)}) = -jA_{\omega}(z^{(0)})^2 \cdot \tilde{\kappa}_{device}(-2\Gamma)
$$ 

This result implies that designing a QPM structure for a desired frequency response is equivalent to designing a spatial profile $\kappa(z)$ whose Fourier transform matches that response.
