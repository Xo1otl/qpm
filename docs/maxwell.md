# Overview

Derivation of **Nonlinear Wave Equation** and **SVEA**.

# Nonlinear Wave Equation

Maxwell's equations:

$$\begin{gather}
\nabla \times \boldsymbol{E} = - \frac{\partial \boldsymbol{B}}{\partial t} \\
\nabla \times \boldsymbol{H} = \frac{\partial \boldsymbol{D}}{\partial t} + \boldsymbol{J} \\
\nabla \cdot \boldsymbol{D} = \rho \\
\nabla \cdot \boldsymbol{B} = 0
\end{gather}$$

Assume $\boldsymbol{J}=0, \rho=0, \boldsymbol{B}=\mu_0 \boldsymbol{H}$. Curl first eq, substitute second:

$$\nabla \times (\nabla \times \boldsymbol{E}) + \mu_0 \epsilon_0 \frac{\partial^2 \boldsymbol{E}}{\partial t^2} = - \mu_0 \frac{\partial^2 \boldsymbol{P}}{\partial t^2}$$

Assume $\nabla \cdot \boldsymbol{E} \approx 0$:

$$
\nabla^2 \boldsymbol{E} - \mu_0 \epsilon_0 \frac{\partial^2 \boldsymbol{E}}{\partial t^2} = \mu_0 \frac{\partial^2 \boldsymbol{P}}{\partial t^2}
$$

$\boldsymbol{P} = \boldsymbol{P}^{(1)} + \boldsymbol{P}_{NL} = \epsilon_0 \chi^{(1)} \boldsymbol{E} + \boldsymbol{P}_{NL}$ and $\epsilon = \epsilon_0 (1 + \chi^{(1)})$:

$$\nabla^2 \boldsymbol{E} - \mu_0 \epsilon \frac{\partial^2 \boldsymbol{E}}{\partial t^2} = \mu_0 \frac{\partial^2 \boldsymbol{P}_{NL}}{\partial t^2}$$

# SVEA

Quasi-monochromatic wave ($\Psi$ transverse, $A$ amplitude):

$$
\boldsymbol{E}_n(\boldsymbol{r}, t) = A_n(z) \Psi_n(x,y) e^{i(\beta_n z - \omega_n t)} \hat{\boldsymbol{e}}_n + c.c.
$$

Normalization:

$$\iint_{-\infty}^{\infty} |\Psi_n(x,y)|^2 dx dy = 1$$

Nonlinear polarization:

$$\boldsymbol{P}_{NL}(\boldsymbol{r}, t) = \sum_n \boldsymbol{P}_n^{NL}(\boldsymbol{r}) e^{- i \omega_n t} + c.c.$$

Substitute into wave eq; $\nabla^2 = \nabla_\perp^2 + \partial^2/\partial z^2$:

$$\left( \frac{d^2 A_n}{dz^2} + 2i\beta_n \frac{d A_n}{dz} - \beta_n^2 A_n \right) \Psi_n e^{i\beta_n z} + A_n (\nabla_\perp^2 \Psi_n) e^{i\beta_n z} + \mu_0 \epsilon \omega_n^2 A_n \Psi_n e^{i\beta_n z} = -\mu_0 \omega_n^2 \boldsymbol{P}_n^{NL} \cdot \hat{\boldsymbol{e}}_n$$

$\Psi_n$ satisfies Helmholtz:

Cancels $\nabla_\perp^2, \beta_n^2, \mu_0 \epsilon \omega_n^2$. Apply SVEA $\left| \frac{d^2 A_n}{dz^2} \right| \ll \left| \beta_n \frac{d A_n}{dz} \right|$:

$$
2i\beta_n \frac{d A_n}{dz} \Psi_n e^{i\beta_n z} = -\mu_0 \omega_n^2 \boldsymbol{P}_n^{NL} \cdot \hat{\boldsymbol{e}}_n
$$
Multiply $\Psi_n^*$, integrate cross-section:
$$
\frac{d A_n}{dz} = i \frac{\mu_0 \omega_n^2}{2\beta_n} e^{-i\beta_n z} \iint_{-\infty}^{\infty} \Psi_n^*(x,y) \boldsymbol{P}_n^{NL}(\boldsymbol{r}) \cdot \hat{\boldsymbol{e}}_n dx dy$$
