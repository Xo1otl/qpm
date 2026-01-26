## Inverse Design QPM

Design a QPM structure with a THG output amplitude of $1.8$ or higher.

Given a system operating at wavelength $\lambda = 1.064$ and temperature $T = 70$ with an initial fundamental amplitude of $\sqrt{10}$, determine the sequence of domain widths $w$ that yields a final THG output amplitude $|A_{3\omega}| \ge 1.7$.

The system is governed by the following coupling coefficients:

$$\kappa_{SHG} = \frac{1.5 \times 10^{-5}}{2 / \pi}, \quad \kappa_{SFG} = 2 \kappa_{SHG}$$

Calculate output using `cwes2.simulate_super_step`. 

The block_size parameter must be a divisor of the number of domains and greater than 20.

**Example 1:**

```python
import jax.numpy as jnp
from qpm import cwes2, mgoslt

wl, T = 1.064, 70.
dk1, dk2 = mgoslt.calc_twm_delta_k(wl, wl, T), mgoslt.calc_twm_delta_k(wl, wl/2, T)
w = jnp.tile(jnp.array([jnp.pi/dk1]*2), 300)
signs = jnp.tile(jnp.array([1., -1.]), 300)
k = 1.5e-5/(2/jnp.pi)
b = jnp.array([jnp.sqrt(10.), 0, 0], dtype=jnp.complex64)

print(jnp.abs(cwes2.simulate_super_step(w, signs*k, signs*k*2, dk1, dk2, b, 30)[2]))
```

**Example 2:**

```python
import jax, optax

opt = optax.lbfgs(1.)
st = opt.init(w)

@jax.jit
def scan_step(c, _):
    p, s = c
    g = jax.grad(lambda x: -jnp.abs(cwes2.simulate_super_step(jnp.abs(x), signs*k, signs*k*2, dk1, dk2, b, 30)[2])**2)(p)
    u, ns = opt.update(g, s, p, value=0, grad=g, value_fn=lambda x: 0)
    return (optax.apply_updates(p, u), ns), None

(w, _), _ = jax.lax.scan(scan_step, (w, st), None, length=500)
```

**Constraints**:

* Device length $\le 20,000$.
* Minimum domain width $\ge 1.5$.
