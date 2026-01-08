from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from scipy.special import erf

from qpm import mgoslt


@dataclass
class SimConfig:
    W: float = 48.0  # um
    t_PE: float = 8.0  # h
    D_PE: float = 0.045  # um^2/h
    t_anneal: float = 90.0  # h
    D_x: float = 1.3  # um^2/h
    D_y: float = 1.3 / 1.5  # um^2/h
    L_x: float = 50.0  # um
    L_y: float = 50.0  # um
    T_degC: float = 25.5  # C
    lambda_fw: float = 1.064  # um
    lambda_shw: float = 0.532  # um
    delta_n0_fw: float = 0.012
    delta_n0_shw: float = 0.017
    C0: float = 1.0
    dx: float = 0.4  # um resolution
    dy: float = 0.4  # um resolution


def get_n_dist(C: np.ndarray, wavelength_um: float, T_degC: float, delta_n: float, C0: float) -> np.ndarray:
    """Calculates the refractive index distribution."""
    n_sub_val = mgoslt.sellmeier_n_eff(wavelength_um, T_degC)
    return n_sub_val + delta_n * (C / C0)


def plot_heatmap(data: np.ndarray, x: np.ndarray, y: np.ndarray, title: str, filepath: str) -> None:
    """Plots and saves a heatmap of the data."""
    fig = go.Figure(data=go.Heatmap(z=data.T, x=x, y=y, colorscale="Viridis", colorbar={"title": "Refractive Index"}))
    fig.update_layout(title=title, xaxis_title="Depth x (um)", yaxis_title="Width y (um)", width=800, height=800)
    fig.write_html(filepath)
    print(f"Saved {filepath}")


def calculate_profiles(cfg: SimConfig | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the refractive index profiles for FW and SHW."""
    if cfg is None:
        cfg = SimConfig()

    dx = cfg.dx
    dy = cfg.dy

    # Dimensions
    d_PE = 2 * np.sqrt(cfg.D_PE * cfg.t_PE)
    print(f"Calculated d_PE: {d_PE:.4f} um")

    # Domain
    # dx, dy passed as arguments
    x_axis = np.arange(-cfg.L_x, cfg.L_x + dx / 2, dx)
    y_axis = np.arange(-cfg.L_y, cfg.L_y + dy / 2, dy)
    X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")

    # Time Evolution (Analytical Solution)
    denom_x = 2 * np.sqrt(cfg.D_x * cfg.t_anneal)
    denom_y = 2 * np.sqrt(cfg.D_y * cfg.t_anneal)

    val_x = 0.5 * (erf((d_PE - X) / denom_x) - erf((0 - X) / denom_x))
    val_y = 0.5 * (erf((cfg.W / 2 - Y) / denom_y) - erf((-cfg.W / 2 - Y) / denom_y))

    C_final = cfg.C0 * val_x * val_y

    print("Calculated concentration profile (Analytical).")

    # Index Construction
    print("Constructing Index Distributions...")
    n_fw = get_n_dist(C_final, cfg.lambda_fw, cfg.T_degC, cfg.delta_n0_fw, cfg.C0)
    n_shw = get_n_dist(C_final, cfg.lambda_shw, cfg.T_degC, cfg.delta_n0_shw, cfg.C0)

    return x_axis, y_axis, n_fw, n_shw


def main() -> None:
    cfg = SimConfig()  # Needed for filenames/titles
    x_axis, y_axis, n_fw, n_shw = calculate_profiles()

    # Output
    print("Saving outputs...")

    plot_heatmap(n_fw, x_axis, y_axis, f"Refractive Index Distribution (FW {cfg.lambda_fw} um)", "out/index_fw.html")
    plot_heatmap(n_shw, x_axis, y_axis, f"Refractive Index Distribution (SHW {cfg.lambda_shw} um)", "out/index_shw.html")


if __name__ == "__main__":
    main()
