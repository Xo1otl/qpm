import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", val=True)

import jax.numpy as jnp
import plotly.graph_objects as go
from jax import jit, vmap

from qpm import cwes2, mgoslt

batch_shg_npda = jit(vmap(cwes2.simulate_shg_npda, in_axes=(None, None, 0, None)))


def plot_domain_widths(widths: jax.Array) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=widths))
    fig.update_layout(
        title_text="Domain Widths",
        xaxis_title="Domain Index",
        yaxis_title="Width (μm)",
    )
    fig.show()


num_domains = round(10000 / 7.2 * 2)  # 10mm 7.2 \mu m period -> 2W output
kappa_mag = 1.5e-5 / (2 / jnp.pi)  # 野呂さんの論文に一定周期の実効的な結合係数が 0.15 W^{-1/2} cm^{-1}と書いてある
temperature = 70.0  # Operating temperature (°C)
wl_center = 1.064
dk = mgoslt.calc_twm_delta_k(wl_center, wl_center, temperature)
l_c = jnp.pi / dk
print(2 * l_c)
p_in = 10
b_in = jnp.array(jnp.sqrt(p_in), dtype=jnp.complex64)


# Define uniform grating parameters
widths_uniform = jnp.ones(num_domains) * l_c
kappas_uniform = kappa_mag * jnp.power(-1.0, jnp.arange(num_domains))
wl_range = 0.0025
wls_uniform = jnp.linspace(wl_center - wl_range / 2, wl_center + wl_range / 2, 1000)

# Simulate at 70 degrees (design temperature)
dks_uniform_70 = mgoslt.calc_twm_delta_k(wls_uniform, wls_uniform, 70.0)
b_out_uniform_70 = batch_shg_npda(widths_uniform, kappas_uniform, dks_uniform_70, b_in)
p_out_uniform_70 = jnp.abs(b_out_uniform_70) ** 2

# Simulate at 71 degrees
dks_uniform_71 = mgoslt.calc_twm_delta_k(wls_uniform, wls_uniform, 75.0)
b_out_uniform_71 = batch_shg_npda(widths_uniform, kappas_uniform, dks_uniform_71, b_in)
p_out_uniform_71 = jnp.abs(b_out_uniform_71) ** 2

# Define chirp parameters
chirp_rate = 5.0e-6
indices = jnp.arange(num_domains)
wls_chirp = jnp.linspace(1.04, 1.068, 1000)
dks_chirp = mgoslt.calc_twm_delta_k(wls_chirp, wls_chirp, temperature)
widths_chirp = l_c / jnp.sqrt(1 + 2 * chirp_rate * l_c * indices)
kappas_chirp = kappa_mag * jnp.power(-1.0, indices)
b_out_chirp = batch_shg_npda(widths_chirp, kappas_chirp, dks_chirp, b_in)
p_out_chirp = jnp.abs(b_out_chirp) ** 2

# Plot
plt.figure(figsize=(10, 6))
plt.plot(wls_uniform, p_out_uniform_70, label="70°C")
plt.plot(wls_uniform, p_out_uniform_71, label="75°C")
plt.xlabel(r"Wavelength ($\mu$m)")
plt.ylabel("Output Power (W)")
plt.title("SHG Spectrum")
plt.legend()
plt.grid(visible=True)
plt.savefig("uniform_npda.png")

plt.figure(figsize=(10, 6))
plt.plot(wls_chirp, p_out_chirp)
plt.xlabel(r"Wavelength ($\mu$m)")
plt.ylabel("Output Power (W)")
plt.title("SHG Spectrum")
plt.grid(visible=True)
plt.savefig("chirp_npda.png")
