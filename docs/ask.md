### Code
```
import jax

jax.config.update("jax_enable_x64", val=True)
import jax.numpy as jnp
from plotly import graph_objects as go

from qpm import cwes, mgoslt

batch_simulate = jax.jit(jax.vmap(cwes.simulate_shg_npda, in_axes=(None, None, 0, None)))

NORO_CRR_FACTOR = 1.07 / 2.84 * 100
design_temp = 70.0
kappa_mag = 1.31e-5 / (2 / jnp.pi)
b_initial = jnp.array(1.0 + 0.0j)
wl_start = 1.025
wl_end = 1.035
wls = jnp.linspace(wl_start, wl_end, 500)

dks = mgoslt.calc_twm_delta_k(wls, wls, design_temp)

num_domains = 555
design_wl = 1.031
shg_width = jnp.pi / mgoslt.calc_twm_delta_k(design_wl, design_wl, design_temp)
widths = jnp.full((num_domains,), shg_width)
kappas = kappa_mag * ((-1) ** jnp.arange(num_domains))

amps = batch_simulate(widths, kappas, dks, b_initial)
effs = jnp.abs(amps) ** 2 * NORO_CRR_FACTOR

fig = go.Figure()
fig.add_trace(go.Scatter(x=wls, y=effs, mode="lines"))
fig.update_layout(
    title="SHG Conversion Efficiency vs Wavelength",
    xaxis_title="Wavelength (microns)",
    yaxis_title="SHG Conversion Efficiency",
)
fig.show()
```

### Definitions

Domains: $w_i$, $i \in \{0..N-1\}$
Pairs: $(w_{2n}, w_{2n+1})$, $n \in \{0..(N/2)-1\}$

* **Period ($L_p$):**
    $$L_p = w_{2n} + w_{2n+1}$$

* **Duty Cycle ($D_n$):**
    $$D_n = \frac{w_{2n}}{L_p}$$

---
### Constraints
1.  $L_p = \text{constant}$ (for all $n$)
2.  `kappa_mag` = constant
3.  $D_n = \text{variable}$ (per $n$)

# Question
Is it possible to broaden SHG spectrum around Fc by optimizing D_n?

**Answer:**
Yes, it is possible. By modulating the duty cycle $D_n$ along the propagation direction $z$, we can effectively modulate the effective nonlinearity $\kappa_{\text{eff}}(z)$.
Since the spectral response is related to the Fourier transform of the spatial nonlinearity profile $\kappa(z)$, we can design a specific $\kappa(z)$ (via apodization or phase modulation) that corresponds to a broader spectrum (e.g., a Gaussian spectrum) and map this profile to the duty cycles $D_n$.
This is a form of inverse design where we define the target spectrum and retrieve the required spatial structure.
