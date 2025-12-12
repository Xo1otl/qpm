# Introduction

When optimizing high-dimensional parameter vectors in a complex loss landscape, standard gradient-based methods like L-BFGS can converge to suboptimal local minima. These solutions are often characterized by being "jagged," sensitive to initialization, and physically less plausible or robust. This document outlines several techniques to alter the search dynamics or the effective loss landscape to favor the discovery of smoother, more desirable solutions.

# Overview

The primary strategies to address this challenge fall into three categories: **Regularization**, which modifies the loss function to penalize non-smooth solutions; **Reparameterization**, which changes the basis of the optimization variables to inherently enforce smoothness; and **Search Strategy Modification**, which alters the optimization algorithm's behavior to better navigate the complex landscape.

# Regularization

Regularization involves adding a penalty term to the loss function. This term quantifies the "jaggedness" of the parameter vector, making non-smooth solutions less attractive to the optimizer.

### Total Variation (TV) Regularization
This method penalizes the absolute difference between adjacent parameter values. It is highly effective at removing noise while preserving sharp edges. The penalty term is `lambda * jnp.sum(jnp.abs(jnp.diff(w)))`, where `lambda` is the regularization strength.

### Tikhonov Regularization (Penalizing Derivatives)
This technique adds a penalty based on the norm of the discrete derivative of the parameter vector.
- **First-order**: Penalizing the first derivative (`jnp.diff(w)`) encourages smoothness.
- **Second-order**: Penalizing the second derivative (`jnp.diff(w, n=2)`) encourages the solution to be not just smooth, but also "flat."
The penalty term is `lambda * jnp.sum(jnp.square(jnp.diff(w, n=...)))`.

# Reparameterization

Instead of optimizing the parameter vector `w` directly, we can represent it using a set of smoother basis functions (e.g., Fourier series, Chebyshev polynomials, or splines). The optimization is then performed on the coefficients of these basis functions. This approach inherently constrains the solution space to smooth functions, effectively removing jagged solutions from consideration. For example, one could define `w` as the output of a function parameterized by a smaller, smoother set of variables `c`: `w = f(c)`.

# Search Strategy Modification

Altering the optimization algorithm and its application can help navigate complex landscapes and avoid sharp, undesirable minima.

### Different Optimizers
While L-BFGS is effective at finding a local minimum quickly, first-order methods like Adam or SGD with Momentum can be better at exploring the loss landscape. Their inherent momentum can help the optimizer "roll past" small, sharp minima and settle into broader, more stable basins.

### Regularization Annealing
This strategy involves starting the optimization with a high regularization strength (`lambda`) to find a very smooth, approximate solution. The value of `lambda` is then gradually decreased over the course of the optimization. This allows the optimizer to first find the general shape of a good solution and then refine its details on a smoother effective loss surface, preventing it from getting trapped in poor local minima early on.

### Coarse-to-Fine Optimization
This approach begins by optimizing a lower-dimensional (coarser) representation of the problem. For instance, in QPM grating design, one could start by optimizing a structure with fewer domains. The resulting smooth solution is then up-sampled and used as the initialization for a higher-resolution optimization. This provides a good starting point that is already in a smooth region of the solution space.
