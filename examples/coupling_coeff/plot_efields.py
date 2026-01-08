import pathlib
from dataclasses import dataclass

import modesolver
import numpy as np
import plotly.graph_objects as go
import scipy.io
from modesolver.postprocess import efields
from modesolver.postprocess.poynting import poynting
from plotly.subplots import make_subplots
from scipy.signal import find_peaks


@dataclass
class Config:
    """Simulation configuration parameters."""

    dx: float = 0.01
    dy: float = 0.01
    wl_fw: float = 0.8
    n_modes_fw: int = 5
    n_modes_shw: int = 15
    material_file_fw: str = "Permittivity_Mesh_fw.mat"
    material_file_shw: str = "Permittivity_Mesh_shw.mat"


@dataclass
class ModeData:
    """Data for a specific waveguide mode."""

    wavelength: float
    neff: float
    mx: int
    my: int
    ey: np.ndarray
    eps_y: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    hz: np.ndarray


def load_materials(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Loads material data from a .mat file and transposes for solver."""
    data = scipy.io.loadmat(str(path))
    # Solver expects (Y, X), data is (X, Y) -> Transpose
    return {
        "EpsX": data["EpsX"].T,
        "EpsY": data["EpsY"].T,
        "EpsZ": data["EpsZ"].T,
    }


def _get_polarization(hx: np.ndarray, hy: np.ndarray) -> int:
    """Determines polarization: 2 for TM, 1 for TE."""
    mag_hx = np.sum(np.abs(hx) ** 2)
    mag_hy = np.sum(np.abs(hy) ** 2)
    return 2 if mag_hx > mag_hy else 1


def _estimate_mode_indices(field: np.ndarray) -> tuple[int, int]:
    """Estimates mode indices (mx, my) from field profile."""
    abs_field = np.abs(field)
    y_peak, x_peak = np.unravel_index(np.argmax(abs_field), abs_field.shape)

    profile_x = abs_field[y_peak, :]
    profile_y = abs_field[:, x_peak]

    threshold = 0.1 * np.max(abs_field)
    peaks_x, _ = find_peaks(profile_x, height=threshold)
    peaks_y, _ = find_peaks(profile_y, height=threshold)

    return len(peaks_x), len(peaks_y)


def calculate_overlap(cfg: Config, mode_fw: ModeData, mode_shw: ModeData, e2yy: float) -> float:
    """Calculates the overlap integral (coupling coefficient)."""
    ey_fw = mode_fw.ey
    ey_shw = mode_shw.ey
    eps_shw = mode_shw.eps_y

    ey_shw_abs = np.abs(ey_shw)

    # Constants
    d33 = 6.3e-6
    e0 = 8.854e-18
    c = 2.99792458e14

    core_mask = np.isclose(eps_shw, e2yy, atol=1e-4)
    term = core_mask * (ey_fw**2) * ey_shw_abs
    sz = d33 * np.real(term)

    da = cfg.dx * cfg.dy
    ol_lin = np.sum(sz) * da

    return np.abs((2 * 2 * np.pi * c / cfg.wl_fw) * e0 * ol_lin / 4) * 1e4


def solve_for_target_mode(  # noqa: PLR0913
    cfg: Config,
    materials: dict[str, np.ndarray],
    wavelength: float,
    n_modes: int,
    target_mx: int,
    target_my: int,
) -> ModeData | None:
    """Runs modesolver and searches for a specific target mode."""
    print(f"Solving for wl={wavelength:.2f}, Target Mode=({target_mx},{target_my})...")

    eps_x = materials["EpsX"]
    eps_y = materials["EpsY"]
    eps_z = materials["EpsZ"]

    neffs, hxs, hys, hzjs = modesolver.wgmodes(
        wavelength,
        3.0,
        n_modes,
        cfg.dx,
        cfg.dy,
        "0000",
        epsxx=eps_x,
        epsyy=eps_y,
        epszz=eps_z,
    )

    for m in range(n_modes):
        hx = hxs[:, :, m]
        hy = hys[:, :, m]
        hzj = hzjs[:, :, m]
        neff = neffs[m]

        if _get_polarization(hx, hy) == 2:  # TM
            mx, my = _estimate_mode_indices(hx)

            if mx == target_mx and my == target_my:
                print(f"Found Mode {m}: neff={neff:.4f}, Index=({mx},{my})")

                # Calculate fields
                ex, ey, ezj = efields(neff, hx, hy, hzj, wavelength, cfg.dx, cfg.dy, epsxx=eps_x, epsyy=eps_y, epszz=eps_z)
                _, _, sz_field = poynting(ex, ey, ezj, hx, hy, hzj)

                # Normalize Field
                power = np.sum(sz_field) * cfg.dx * cfg.dy
                norm_factor = np.sqrt(376.73 / power)
                ey_norm = ey * norm_factor

                return ModeData(
                    wavelength=wavelength,
                    neff=neff,
                    mx=mx,
                    my=my,
                    ey=ey_norm,
                    eps_y=eps_y,
                    hx=hx,
                    hy=hy,
                    hz=hzj,
                )

    return None


def visualize_fields(mode_fw: ModeData, mode_shw: ModeData, materials_fw: dict[str, np.ndarray]) -> None:
    """Visualizes the permittivity and normalized E-fields using Plotly."""
    eps_x_fw = materials_fw["EpsX"]

    # Normalize fields for visualization
    ey_fw_norm_val = np.max(np.abs(mode_fw.ey))
    ey_shw_norm_val = np.max(np.abs(mode_shw.ey))

    ey_fw_plot = np.abs(mode_fw.ey) / ey_fw_norm_val
    ey_shw_plot = np.abs(mode_shw.ey) / ey_shw_norm_val

    # Create subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Permittivity EpsX (FW)", "Normalized |Ey| FW", "Normalized |Ey| SHW"))

    # 1. Permittivity
    fig.add_trace(go.Heatmap(z=np.real(eps_x_fw), colorscale="Greys", name="EpsX"), row=1, col=1)

    # 2. FW Field
    fig.add_trace(go.Heatmap(z=ey_fw_plot, colorscale="Jet", name="|Ey| FW"), row=1, col=2)

    # 3. SHW Field
    fig.add_trace(go.Heatmap(z=ey_shw_plot, colorscale="Jet", name="|Ey| SHW"), row=1, col=3)

    fig.update_layout(title_text="Waveguide Modes and Permittivity", height=500, width=1200, showlegend=False)

    out_file = "./out/fields_visualization.html"
    fig.write_html(out_file)
    print(f"Visualization saved to {out_file}")


def main() -> None:
    cfg = Config()
    script_dir = pathlib.Path(__file__).parent

    # Load Materials
    mat_fw = load_materials(script_dir / cfg.material_file_fw)
    mat_shw = load_materials(script_dir / cfg.material_file_shw)

    # Solve FW
    mode_fw = solve_for_target_mode(cfg, mat_fw, cfg.wl_fw, cfg.n_modes_fw, target_mx=1, target_my=1)
    if not mode_fw:
        print("Error: FW TM00 Mode (1,1) not found.")
        return

    # Solve SHW
    mode_shw = solve_for_target_mode(cfg, mat_shw, cfg.wl_fw / 2, cfg.n_modes_shw, target_mx=1, target_my=4)
    if not mode_shw:
        print("Error: SHW TM03 Mode (1,4) not found.")
        return

    # Visualize
    visualize_fields(mode_fw, mode_shw, mat_fw)

    # Calculate Overlap
    e2yy = np.max(mode_shw.eps_y)

    kappa = calculate_overlap(cfg, mode_fw, mode_shw, e2yy)
    print(f"\nOverlap Integral (kappa): {kappa:.4f}")


if __name__ == "__main__":
    main()
