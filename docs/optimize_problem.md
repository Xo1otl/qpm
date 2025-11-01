**Objective:** Maximize the magnitude of the following complex integral:

$$ \max_{\kappa(z)} \left| \int_0^L \kappa(z) e^{-i\Delta k z} dz \right| $$

**Constraint:** The function $\kappa(z)$ is restricted to take one of two values at any point $z$:

$$ \kappa(z) = \pm \kappa_0 $$

where $\kappa_0$ is a positive constant.

**Answer:** The optimal solution *must* be of the form:

$$ \kappa^*(z) = \kappa_0 \cdot \text{sign}(\cos(\Delta k z + \phi^*)) $$

**Objective:** Find the control function $\kappa^*(z)$ that solves the following optimization problem:
$$ \max_{\kappa(z)} |S[\kappa]| $$

The functional $S[\kappa]$ is defined as:
$$ S[\kappa] = \int_0^L dz_2 \int_0^{z_2} dz_1 \, \kappa(z_1) \kappa(z_2) e^{-i(\Delta k_1 z_1 + \Delta k_2 z_2)} $$

**Constraint:**
The control function $\kappa(z)$ is restricted to:
$$ \kappa(z) \in \{+\kappa_0, -\kappa_0\} \quad \forall z \in [0, L] $$
where $\kappa_0$ is a positive real constant.

**Task:**
Determine the structure of the optimal control function $\kappa^*(z)$.
