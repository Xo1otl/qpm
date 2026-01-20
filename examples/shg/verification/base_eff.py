import jax.numpy as jnp

from qpm import cwes2, mgoslt

delta_k = mgoslt.calc_twm_delta_k(1.064, 1.064, 70)
period = 2 * jnp.pi / delta_k
n_periods = int(15000.0 / period)
domain_widths = jnp.tile(jnp.array([period / 2, period / 2]), n_periods)
kappa_vals = jnp.tile(jnp.array([1.5e-5, -1.5e-5]), n_periods)
P_omega = 10  # [W]

b_initial = jnp.array(jnp.sqrt(P_omega))
_, amp_trace = cwes2.simulate_shg_npda_trace(domain_widths, kappa_vals, delta_k, b_initial)

P_2omega = jnp.abs(amp_trace[-1]) ** 2
print(f"Input power: {P_omega:.4e} [W], Output Power: {P_2omega:.4e} [W]")
