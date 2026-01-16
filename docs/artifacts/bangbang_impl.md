# Procedure
1.  **Initialize** $\kappa(z)$ (e.g., randomly).
2.  **Calculate $S[\kappa]$**: Compute the functional $S$ using the current $\kappa$.
3.  **Update $\phi$**: Set the phase $\phi = -\arg S[\kappa]$.
4.  **Calculate $w(z)$**: Compute the switching function $w_{\phi}(z;\kappa)$ using the current $\kappa$ and the updated $\phi$.
5.  **Update $\kappa$**: Determine the new control $\kappa_{\text{new}}(z) = \kappa_0 \operatorname{sign} w(z)$.
6.  **Repeat steps 2-5** until $\kappa$ converges (i.e., $\kappa_{\text{new}} = \kappa$).
