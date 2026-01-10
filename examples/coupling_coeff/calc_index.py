from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from scipy.special import erf

from qpm import mgoslt


@dataclass
class SimConfig:
    W: float = 48.0  # um (Width)
    t_PE: float = 8.0  # h
    D_PE: float = 0.045  # um^2/h
    t_anneal: float = 90.0  # h
    D_depth: float = 1.3  # um^2/h (Vertical/Y)
    D_width: float = 1.3 / 1.5  # um^2/h (Transverse/X)
    L_depth: float = 50.0  # um (Y limit)
    L_width: float = 50.0  # um (X limit)
    T_degC: float = 25.5  # C
    lambda_fw: float = 1.064  # um
    lambda_shw: float = 0.532  # um
    delta_n0_fw: float = 0.012
    delta_n0_shw: float = 0.017
    C0: float = 1.0
    dx: float = 0.2  # um resolution (Width step)
    dy: float = 0.2  # um resolution (Depth step)


def get_n_dist(C: np.ndarray, wavelength_um: float, T_degC: float, delta_n: float, C0: float) -> np.ndarray:
    """Calculates the refractive index distribution."""
    n_sub_val = mgoslt.sellmeier_n_eff(wavelength_um, T_degC)
    return n_sub_val + delta_n * (C / C0)


def plot_heatmap(data: np.ndarray, x: np.ndarray, y: np.ndarray, title: str, filepath: str) -> None:
    """Plots and saves a heatmap of the data.
    data: (Width, Depth) array ?
    Actually Plotly Heatmap expects z as (y, x) or (rows, cols).
    If we stick to standard: x-axis is Width, y-axis is Depth.
    Data should be z[y, x] or similar.
    We will strictly pass x (Width axis) and y (Depth axis).
    """
    # z expects (rows=y, cols=x). If data is (Width, Depth) -> (cols, rows).
    # So we should pass data.T (Depth, Width) which maps to (y, x).
    fig = go.Figure(data=go.Heatmap(z=data.T, x=x, y=y, colorscale="Viridis", colorbar={"title": "Refractive Index"}))
    fig.update_layout(title=title, xaxis_title="Width x (um)", yaxis_title="Depth y (um)", width=800, height=800)
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
    # x: Width, y: Depth
    x_axis = np.arange(-cfg.L_width, cfg.L_width + dx / 2, dx)
    y_axis = np.arange(-cfg.L_depth, cfg.L_depth + dy / 2, dy)

    # Meshgrid: X (Width), Y (Depth)
    X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")

    # Time Evolution (Analytical Solution)
    # D_width corresponds to diffusion in Width (X)
    # D_depth corresponds to diffusion in Depth (Y)
    denom_width = 2 * np.sqrt(cfg.D_width * cfg.t_anneal)
    denom_depth = 2 * np.sqrt(cfg.D_depth * cfg.t_anneal)

    # Initial Condition:
    # Defined on |x| <= W/2 and 0 <= y <= d_PE

    # Diffusion in Width (X):
    # Initial block width W centered at 0.
    val_x = 0.5 * (erf((cfg.W / 2 - X) / denom_width) - erf((-cfg.W / 2 - X) / denom_width))

    # Diffusion in Depth (Y):
    # Initial block depth d_PE starting at 0.
    # We treat surface at y=0.
    val_y = 0.5 * (erf((d_PE - Y) / denom_depth) - erf((0 - Y) / denom_depth))

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
