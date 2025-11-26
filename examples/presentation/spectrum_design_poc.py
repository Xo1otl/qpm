import jax
import jax.numpy as jnp
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from qpm import cwes, mgoslt

# Enable x64
jax.config.update("jax_enable_x64", val=True)

# --- 1. Common Simulation Parameters ---
NORO_CRR_FACTOR = 1.07 / 2.84 * 100
design_temp = 70.0
kappa_mag = 1.31e-5 / (2 / jnp.pi)
wl_start = 1.025
wl_end = 1.037
wls = jnp.linspace(wl_start, wl_end, 1000)

# Calculate phase mismatch
dks = mgoslt.calc_twm_delta_k(wls, wls, design_temp)
b_initial = jnp.array(1.0 + 0.0j)
batch_simulate = jax.jit(jax.vmap(cwes.simulate_shg_npda, in_axes=(None, None, 0, None)))

# --- 2. Target Profile Generation (Common) ---
# Using the same target as provided in the snippet
num_periods = 10000  # Adjusted for period-based logic
design_wl = 1.031
shg_width = jnp.pi / mgoslt.calc_twm_delta_k(design_wl, design_wl, design_temp)
Lp = 2 * shg_width  # The full period Lambda
L_total = num_periods * Lp

z_period_centers = (jnp.arange(num_periods) + 0.5) * Lp
z_center = L_total / 2.0
z_n = z_period_centers - z_center

# Target: Sinc * Gaussian
apodization_scale_factor = 30.0
sigma = (L_total / 2.0) * 0.5
sinc_arg = z_n * apodization_scale_factor / (L_total / 2.0)
target_profile = jnp.sinc(sinc_arg) * jnp.exp(-(z_n**2) / (2 * sigma**2))

# Normalize
max_val = jnp.max(jnp.abs(target_profile))
norm_profile = target_profile / max_val

# Extract D and Sign
mag_profile = jnp.abs(norm_profile)
sign_profile = jnp.sign(norm_profile)
D_n = jnp.arcsin(mag_profile) / jnp.pi

# --- 3. Geometry Construction ---

# A. Left-Anchored (2-Domain) - The "Control"
# ---------------------------------------------
w_left_1 = D_n * Lp
w_left_2 = (1 - D_n) * Lp

# Interleave [w1, w2, w1, w2...]
widths_2d = jnp.column_stack((w_left_1, w_left_2)).ravel()

# Kappas: (+, -) pattern multiplied by profile sign
# If sign is +, period is (+, -). If sign is -, period is (-, +).
base_signs_2d = jnp.tile(jnp.array([1.0, -1.0]), num_periods)
# Repeat profile sign for both domains in a period
period_signs_repeated = jnp.repeat(sign_profile, 2)
kappas_2d = kappa_mag * base_signs_2d * period_signs_repeated

# B. Center-Anchored (3-Domain) - The "Proposal"
# ----------------------------------------------
# Structure: [Gap_Left, Pulse, Gap_Right]
# Gap_Left  = (1 - D) * Lp / 2
# Pulse     = D * Lp
# Gap_Right = (1 - D) * Lp / 2

gap_widths = (1 - D_n) * Lp / 2.0
pulse_widths = D_n * Lp

# Interleave [gap, pulse, gap, gap, pulse, gap...]
widths_3d = jnp.column_stack((gap_widths, pulse_widths, gap_widths)).ravel()

# Kappas: (-, +, -) pattern multiplied by profile sign
# Note: To match the standard QPM global phase, we use [-1, 1, -1]
# so the 'Pulse' is the +k domain.
base_signs_3d = jnp.tile(jnp.array([-1.0, 1.0, -1.0]), num_periods)
period_signs_repeated_3d = jnp.repeat(sign_profile, 3)
kappas_3d = kappa_mag * base_signs_3d * period_signs_repeated_3d


# --- 4. Execution ---
print("Simulating Left-Anchored (2-Domain)...")
amps_2d = batch_simulate(widths_2d, kappas_2d, dks, b_initial)
effs_2d = jnp.abs(amps_2d) ** 2 * NORO_CRR_FACTOR

print("Simulating Center-Anchored (3-Domain)...")
amps_3d = batch_simulate(widths_3d, kappas_3d, dks, b_initial)
effs_3d = jnp.abs(amps_3d) ** 2 * NORO_CRR_FACTOR

# --- 5. Visualization & Analysis ---
fig = make_subplots(rows=2, cols=1, subplot_titles=("Spectral Shape Comparison", "Error/Asymmetry Analysis"), vertical_spacing=0.15)

# Plot 1: Spectra
fig.add_trace(
    go.Scatter(x=wls, y=effs_2d, mode="lines", name="2-Domain (Left-Anchored)", line={"color": "red", "width": 2, "dash": "dot"}),
    row=1,
    col=1,
)
fig.add_trace(go.Scatter(x=wls, y=effs_3d, mode="lines", name="3-Domain (Center-Anchored)", line={"color": "blue", "width": 2}), row=1, col=1)

# Plot 2: Asymmetry Check (Difference from flipped version)
# We flip the spectrum around the center wavelength index to check symmetry
mid_idx = len(wls) // 2
# Simple heuristic: visual asymmetry
fig.add_trace(go.Scatter(x=wls, y=effs_2d - jnp.flip(effs_2d), mode="lines", name="Asymmetry (2-Domain)", line={"color": "red"}), row=2, col=1)
fig.add_trace(go.Scatter(x=wls, y=effs_3d - jnp.flip(effs_3d), mode="lines", name="Asymmetry (3-Domain)", line={"color": "blue"}), row=2, col=1)

fig.update_layout(height=800, width=900, title_text="Verification: Center-Anchored Modulation")
fig.update_xaxes(title_text="Wavelength (µm)", row=2, col=1)
fig.update_yaxes(title_text="Efficiency", row=1, col=1)
fig.update_yaxes(title_text="Asymmetry (Delta)", row=2, col=1)

fig.show()
