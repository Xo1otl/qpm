# Overview
**InitialState**(PE) -> **TimeEvolution**(APE) -> **IndexConstruction** -> **ModeSolving** -> **CouplingCalculation**.

# InitialState
Rectangular step function.
* **Depth ($d_{PE}$)**:
  $$ d_{PE} = 2\sqrt{D_{PE} t_{PE}} $$
  Using $D_{PE} = 0.045 \, \mu\text{m}^2/\text{h}$, $t_{PE} = 8 \, \text{h}$:
  $$ d_{PE} = 2\sqrt{0.045 \times 8} = 1.2 \, \mu\text{m} $$
* **Width ($W$)**: $50 \, \mu\text{m}$.
* **Profile**: $C_0$ inside, $0$ outside.

# TimeEvolution
Solve **2D Diffusion Eq**:
$$ \frac{\partial C}{\partial t} = \frac{\partial}{\partial x}\left(D_x \frac{\partial C}{\partial x}\right) + \frac{\partial}{\partial y}\left(D_y \frac{\partial C}{\partial y}\right) $$
* $x$: Depth, $y$: Width.
* **$D$**: $D_x \approx 1.5 D_y$.
* **BC**: $D=0$ at $x=0$.

# IndexConstruction
$$ n(x,y) = n_{\text{sub}} + \Delta n(x,y) $$
$$ \Delta n(x,y) = \Delta n_0 \times \frac{C(x,y)}{C_0} $$
* $\Delta n_0$: Max index change.
* $n_{\text{sub}}$: Sellmeier.

# ModeSolving
Calculate $E(x,y)$ (FEM) for TM$_{00}$:
* $E_{\omega}, E_{2\omega}, E_{3\omega}$.

# CouplingCalculation
### 1. SHG ($\omega + \omega \to 2\omega$)
$$ \kappa_{\text{SHG}} = \frac{\omega_{2}\epsilon_0}{4} \iint [E_{2\omega}]^* d(x,y) [E_{\omega}]^2 \, dx dy $$

### 2. SFG ($\omega + 2\omega \to 3\omega$)
$$ \kappa_{\text{SFG}} = \frac{\omega_3\epsilon_0}{2} \iint [E_{3\omega}]^* d(x,y) E_{2\omega} E_{\omega} \, dx dy $$

# Parameters
| Param | Val | Source |
| :--- | :--- | :--- |
| **Initial Shape** | $50 \times 1.2 \, \mu m$ Rect | p.109, p.107 |
| **$t_{anneal}$** | $100$ h | p.109 |
| **$t_{PE}$** | $8$ h | p.109, p.110 |
| **$D_{PE}$** | $0.045 \mu m^2/h$ | p.107 |
| **$D_x$** | $1.3 \mu m^2/h$ | p.108 |
| **$D_y$** | $1.3 / 1.5 \mu m^2/h$ | p.109 |
| **$\Delta n_0$ (@1030nm)** | $0.012$ | p.109 |
| **$\Delta n_0$ (@532nm)** | $0.017$ | p.111 |
| **$d_{33}$** | $1.38 \times 10^{-5} \mu m/V$ | p.100 |

# TODO
* 355nm $\Delta n_0$ unknown.
* Alt: use $\kappa_{SHG} \to \kappa_{SFG}$ method.

# Outputs
## IndexConstruction
```
--- QPM Index Construction (Refactored) ---
Parameters:
  d_PE (calculated): 1.2000 um
  W: 50.0 um
  t_anneal: 100.0 h
  D_x: 1.3 um^2/h
  D_y: 0.8667 um^2/h
  Temp: 70.0 C

Substrate Indices:
  n_sub(@1.031um): 2.132502
  n_sub(@0.5155um): 2.204083

Peak Index (x=0, y=0) after Annealing:
  n(@1.031um): 2.133173 (Delta: 0.000671)
  n(@0.5155um): 2.205033 (Delta: 0.000950)

Index at x=5.0, y=0.0:
  n(@1.031um): 2.133142
```

# Question
Is this output reasonable for APE waveguide?
