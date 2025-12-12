# Calculation Methodology for $\kappa$

## Overview of the Calculation Process
To accurately model the channel waveguide used for cTHG, a **2D simulation** is required. The process follows a **two-step time-evolution method**:
1.  **Step 1 (Proton Exchange - PE):** Establish the initial H$^+$ concentration profile $C(x,y,0)$ using 1D parameters to define the depth ($d_{PE}$) and the mask design to define the width ($W$).
2.  **Step 2 (Annealing - APE):** numerical calculation of thermal diffusion using the **2D diffusion equation**.
3.  **Step 3 (Mode Solving):** Calculate eigenmodes ($E$) from the resulting 2D refractive index profile.
4.  **Step 4 (Coupling):** Calculate $\kappa_{SHG}$ and $\kappa_{SFG}$ using overlap integrals.

---

## Step 1: Initial State Definition (PE Process)
Before solving the differential equation, the initial H$^+$ distribution (at $t=0$ relative to annealing) is defined as a **rectangular step function**.

*   **Depth ($d_{PE}$)**: Calculated using the 1D diffusion coefficient derived from slab waveguide experiments.
    $$ d_{PE} = 2\sqrt{D_{PE} t_{PE}} $$
*   **Width ($W$)**: Defined by the photomask opening width (**$W = 50 \, \mu\text{m}$** is used in the simulation in Chapter 4).
*   **Initial Profile $C(x,y)_{initial}$**:
    *   $C = C_0$ inside the rectangle defined by depth $d_{PE}$ and width $W$.
    *   $C = 0$ outside this region.

---

## Step 2: 2D Thermal Diffusion (APE Process)
The initial rectangular profile is evolved over time $t_a$ (annealing time) by solving the **2D Diffusion Equation** numerically.

**Equation**:
$$ \frac{\partial C}{\partial t} = \frac{\partial}{\partial x}\left(D_x \frac{\partial C}{\partial x}\right) + \frac{\partial}{\partial y}\left(D_y \frac{\partial C}{\partial y}\right) $$

*   **Coordinates**:
    *   $x$: Depth direction (perpendicular to substrate).
    *   $y$: Width direction (parallel to substrate).
*   **Diffusion Coefficients ($D$):**
    *   $D_x$: Corresponds to $D_a$ (measured in 1D experiments).
    *   $D_y$: Assumed to be smaller due to anisotropy ($D_x \approx 1.5 D_y$ based on literature/assumptions in text).
*   **Boundary Conditions**:
    *   $D = 0$ at the crystal surface ($x=0$), assuming no outward diffusion into the air.

---

## Step 3: Refractive Index Construction
Convert the diffused H$^+$ concentration profile $C(x,y)$ into a refractive index distribution $n(x,y)$ for the relevant wavelengths ($\lambda_\omega, \lambda_{2\omega}, \lambda_{3\omega}$).

$$ n(x,y) = n_{\text{sub}} + \Delta n(x,y) $$
$$ \Delta n(x,y) = \Delta n_0 \times \frac{C(x,y)}{C_0} $$

*   $\Delta n_0$: Maximum index change (experimentally determined).
*   $n_{\text{sub}}$: Substrate index from Sellmeier equations.

---

## Step 4: Optical Mode Analysis
Solve for the transverse electric field distributions $E(x,y)$ using a mode solver (Finite Element Method).

*   **Target Modes**: Fundamental (TM$_{00}$) modes for all three waves involved in cTHG.
    *   Fundamental: $E_{\omega}(x,y)$
    *   Second Harmonic: $E_{2\omega}(x,y)$
    *   Third Harmonic: $E_{3\omega}(x,y)$

---

## Step 5: Coupling Coefficient Calculation
Calculate the overlap integrals. The nonlinear coefficient $d(x,y)$ is determined by the domain inversion structure (periodically poled).

### 1. SHG Coupling ($\omega + \omega \to 2\omega$)
$$ \kappa_{\text{SHG}} = \frac{\omega_{2}\epsilon_0}{2} \iint [E_{2\omega}]^* d(x,y) [E_{\omega}]^2 \, dx dy $$
*(Note: Eq 2.45 in the text uses a slightly different pre-factor arrangement, but the physical integral is the overlap of the SHG field with the square of the fundamental field).*

### 2. SFG Coupling ($\omega + 2\omega \to 3\omega$)
$$ \kappa_{\text{SFG}} = \frac{\omega_3\epsilon_0}{2} \iint [E_{3\omega}]^* d(x,y) E_{2\omega} E_{\omega} \, dx dy $$
*(Based on Eq 2.77, representing the sum frequency generation).*

---

## Summary of Key Parameters

| パラメータ | 値 | ソース (PDFページ) |
| :--- | :--- | :--- |
| **初期形状** | 幅 $50 \mu m$ × 深さ $1.2 \mu m$ の矩形 | p.109, p.107(計算) |
| **拡散時間 ($t_{anneal}$)** | $100$ h | p.109 |
| **PE時間 ($t_{PE}$)** | $8$ h | p.109, p.110 |
| **拡散係数 $D_x$** | $1.3 \mu m^2/h$ | p.108 |
| **拡散係数 $D_y$** | $1.3 / 1.5 \mu m^2/h$ | p.109 |
| **$\Delta n_0$ (@1030nm)** | $0.012$ | p.109 |
| **$\Delta n_0$ (@532nm)** | $0.017$ | p.111 |
| **非線形定数 $d_{33}$** | $1.38 \mu m/V$ | p.100 |

## TODO
355nmに対する\Delta n_0の値がわからない (E_3の計算に必要)
