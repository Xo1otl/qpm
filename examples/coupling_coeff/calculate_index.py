import jax.numpy as jnp
from jax.scipy.special import erf
import numpy as np
import plotly.graph_objects as go
from qpm import mgoslt

# Constants and Parameters
TEMP = 70.0  # Celsius

# Initial State parameters
D_PE = 0.045  # um^2/h
t_PE = 8.0  # h
# d_PE = 2 * sqrt(D_PE * t_PE)
d_PE = 2 * jnp.sqrt(D_PE * t_PE)  # Expected 1.2 um
W = 50.0  # um

# Annealing parameters
t_anneal = 100.0  # h
D_x = 1.3  # um^2/h
D_y = D_x / 1.5  # um^2/h

# Refractive Index parameters
DELTA_N0_MAP = {
    # Approx wavelength (um) -> Delta n0
    1.03: 0.012,
    0.515: 0.017,  # Using 532nm value for SHG (~515nm)
}


def get_delta_n0(wl_um):
    """
    Returns delta_n0 for a given wavelength.
    Simple nearest neighbor lookup for the provided scalar values.
    """
    # Keys are 1.03 and 0.515.
    # If wl is closer to 1.03, return 0.012
    # If wl is closer to 0.515, return 0.017

    dist_fund = jnp.abs(wl_um - 1.031)
    dist_sh = jnp.abs(wl_um - 0.5155)

    # We can use jax.lax.cond or just python control flow if wl is a scalar float
    # Assuming wl is a python float or scalar array
    return jnp.where(dist_fund < dist_sh, 0.012, 0.017)


def concentration_distribution(x, y, t_diff):
    """
    Calculates the normalized concentration C(x,y)/C0 after diffusion time t_diff.

    Physics:
    - Initial profile: Rectangular block of width W (y) and depth d_PE (x).
    - Boundary Condition: Reflecting (Neumann) at x=0.
    - Domain: Semi-infinite x >= 0, Infinite y.

    Solution using product of 1D solutions (superposition/separation of variables):
    C(x,y,t) = Cx(x,t) * Cy(y,t)

    Cx(x,t): Source -d_PE to d_PE (due to image source).
    Cx = 0.5 * [erf((d_PE - x)/2sqrt(Dt)) + erf((d_PE + x)/2sqrt(Dt))]

    Cy(y,t): Source -W/2 to W/2.
    Cy = 0.5 * [erf((W/2 - y)/2sqrt(Dt)) + erf((W/2 + y)/2sqrt(Dt))]
    """

    # Vertical Diffusion (x direction)
    # diff_len_x = 2 * sqrt(D_x * t)
    Lx = 2 * jnp.sqrt(D_x * t_diff)
    Cx = 0.5 * (erf((d_PE - x) / Lx) + erf((d_PE + x) / Lx))

    # Horizontal Diffusion (y direction)
    Ly = 2 * jnp.sqrt(D_y * t_diff)
    Cy = 0.5 * (erf((W / 2 - y) / Ly) + erf((W / 2 + y) / Ly))

    return Cx * Cy


def get_index_profile(x, y, wl, temp):
    """
    Calculates the refractive index n(x,y) at a specific wavelength and temperature.
    n(x,y) = n_sub(wl, T) + delta_n0(wl) * (C(x,y)/C0)
    """
    n_sub = mgoslt.sellmeier_n_eff(wl, temp)
    delta_n0 = get_delta_n0(wl)
    C_norm = concentration_distribution(x, y, t_anneal)
    return n_sub + delta_n0 * C_norm


def plot_index_profile(wl, temp, filename="index_profile.html"):
    """
    Generates a heatmap of the index profile using Plotly.
    x-axis: Width (y parameter)
    y-axis: Depth (x parameter)
    """
    # Define grid
    # Depth x: 0 to 40 um
    x_depth = jnp.linspace(0, 40, 200)
    # Width y: -40 to 40 um
    y_width = jnp.linspace(-40, 40, 200)

    # Create meshgrid
    # We want Z[row, col] where row is depth, col is width
    Y_width, X_depth = jnp.meshgrid(y_width, x_depth)

    Z = get_index_profile(X_depth, Y_width, wl, temp)

    # Convert to numpy for Plotly
    Z_np = np.array(Z)
    x_ax = np.array(y_width)
    y_ax = np.array(x_depth)

    fig = go.Figure(data=go.Heatmap(z=Z_np, x=x_ax, y=y_ax, colorscale="Viridis", colorbar=dict(title="Refractive Index"), reversescale=False))

    fig.update_layout(
        title=f"Refractive Index Distribution @ {wl} um, {temp}°C",
        xaxis_title="Width (µm)",
        yaxis_title="Depth (µm)",
        yaxis=dict(autorange="reversed"),  # Depth increases downwards
        width=800,
        height=600,
    )

    # fig.write_html(filename)
    fig.show()
    print(f"Plot saved to {filename}")


def main():
    print("--- QPM Index Construction ---")

    # Wavelengths
    wl_fund = 1.031
    wl_sh = wl_fund / 2.0

    print(f"Parameters:")
    print(f"  d_PE: {d_PE:.4f} um")
    print(f"  W: {W} um")
    print(f"  t_anneal: {t_anneal} h")
    print(f"  D_x: {D_x} um^2/h")
    print(f"  D_y: {D_y:.4f} um^2/h")
    print(f"  Temp: {TEMP} C")

    # Calculate Substrate Indices
    n_sub_fund = mgoslt.sellmeier_n_eff(wl_fund, TEMP)
    n_sub_sh = mgoslt.sellmeier_n_eff(wl_sh, TEMP)

    print(f"\nSubstrate Indices:")
    print(f"  n_sub(@{wl_fund}um): {n_sub_fund:.6f}")
    print(f"  n_sub(@{wl_sh:.4f}um): {n_sub_sh:.6f}")

    # Verify Center Index (Max Index)
    # Expected: n_sub + delta_n0 * 1 (approx, if t_anneal is small or source is large)
    # Actually with t=100h, diffusion is significant.
    # d_PE = 1.2 um. Lx = 2*sqrt(1.3*100) = 2*sqrt(130) ~ 22.8 um.
    # d_PE / Lx ~ 0.05. erf(0.05) is small. So peak concentration will drop significantly.

    val_fund_0 = get_index_profile(0.0, 0.0, wl_fund, TEMP)
    val_sh_0 = get_index_profile(0.0, 0.0, wl_sh, TEMP)

    print(f"\nPeak Index (x=0, y=0) after Annealing:")
    print(f"  n(@{wl_fund}um): {val_fund_0:.6f} (Delta: {val_fund_0 - n_sub_fund:.6f})")
    print(f"  n(@{wl_sh:.4f}um): {val_sh_0:.6f} (Delta: {val_sh_0 - n_sub_sh:.6f})")

    # Sample grid point
    x_sample = 5.0
    y_sample = 0.0
    val_fund_sample = get_index_profile(x_sample, y_sample, wl_fund, TEMP)
    print(f"\nIndex at x={x_sample}, y={y_sample}:")
    print(f"  n(@{wl_fund}um): {val_fund_sample:.6f}")

    # Plotting
    plot_filename = "index_distribution_fund.html"
    print(f"\nGenerating plot for Fundamental wave to {plot_filename}...")
    plot_index_profile(wl_fund, TEMP, plot_filename)


if __name__ == "__main__":
    main()
