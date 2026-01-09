from dataclasses import dataclass

import modesolver
import numpy as np
import plotly.graph_objects as go
import scipy.io
from calc_index import SimConfig, calculate_profiles
from modesolver.postprocess import efields
from modesolver.postprocess.poynting import poynting


@dataclass
class ModeData:
    """Data for a specific waveguide mode."""

    wavelength: float
    neff: float
    mx: int
    my: int
    ex: np.ndarray  # Dominant component (Depth)
    ey: np.ndarray  # Transverse component (Width)
    ez: np.ndarray  # Longitudinal component
    eps: np.ndarray  # Refractive index squared (n^2)
    hx: np.ndarray
    hy: np.ndarray
    hz: np.ndarray


def prepare_materials(n_dist: np.ndarray) -> dict[str, np.ndarray]:
    """Prepares material dictionary from refractive index distribution."""
    # Solver expects (Rows=Y, Cols=X).
    # n_dist is (X, Y).
    # We transpose so that Solver-Rows correspond to Width.
    eps = n_dist**2
    return {
        "EpsX": eps.T,
        "EpsY": eps.T,
        "EpsZ": eps.T,
    }


def _get_polarization(hx: np.ndarray, hy: np.ndarray) -> int:
    """
    Determines polarization based on dominant magnetic field component.
    Returns: 2 for TM (Ex dominant), 1 for TE (Ey dominant).
    """
    mag_hx = np.sum(np.abs(hx) ** 2)
    mag_hy = np.sum(np.abs(hy) ** 2)
    return 2 if mag_hy > mag_hx else 1


def plot_field(mode: ModeData, out_path: str) -> None:
    """Plots and saves a heatmap of the dominant electric field Ex."""
    fig = go.Figure(data=go.Heatmap(z=np.abs(mode.ex), colorscale="Viridis", colorbar={"title": "Electric Field |Ex|"}))
    fig.update_layout(
        title=f"TM{mode.mx - 1}{mode.my - 1} Mode Field Distribution (wl={mode.wavelength} um)",
        xaxis_title="Width y (approx pixels)",
        yaxis_title="Depth x (approx pixels)",
    )
    fig.write_html(out_path)
    print(f"Saved {out_path}")


def save_mode_data(mode: ModeData, filename: str) -> None:
    """Saves complete mode data (Ex, Ey, Ez, H-fields, etc) to a .mat file."""
    data = {
        "wavelength": mode.wavelength,
        "neff": mode.neff,
        "mx": mode.mx,
        "my": mode.my,
        "ex": mode.ex,
        "ey": mode.ey,
        "ez": mode.ez,
        "eps": mode.eps,
        "hx": mode.hx,
        "hy": mode.hy,
        "hz": mode.hz,
    }
    scipy.io.savemat(filename, data)
    print(f"Saved complete mode data to {filename}")


def solve_for_tm00(
    cfg: SimConfig,
    materials: dict[str, np.ndarray],
    wavelength: float,
    n_modes: int = 1,
) -> ModeData | None:
    print(f"Solving for wl={wavelength:.3f} um, searching for fundamental TM (Ex-dominant) mode...")

    eps_x = materials["EpsX"]
    eps_y = materials["EpsY"]
    eps_z = materials["EpsZ"]

    # Run Solver
    neffs, hxs, hys, hzjs = modesolver.wgmodes(
        wavelength,
        2.2,  # Guess index
        n_modes,
        cfg.dx,
        cfg.dy,
        "0000",  # Boundary conditions
        epsxx=eps_x,
        epsyy=eps_y,
        epszz=eps_z,
    )

    # Ensure arrays are 3D
    if hxs.ndim == 2:
        hxs = hxs[:, :, np.newaxis]
        hys = hys[:, :, np.newaxis]
        hzjs = hzjs[:, :, np.newaxis]
        neffs = np.atleast_1d(neffs)

    for m in range(n_modes):
        hx = hxs[:, :, m]
        hy = hys[:, :, m]
        neff = neffs[m]

        if _get_polarization(hx, hy) == 2:  # Found Ex dominant mode (TM)
            print(f"Found Fundamental TM Mode (Ex dominant): neff={neff:.4f} (Mode index {m})")

            hzj = hzjs[:, :, m]

            # Calculate raw E-fields (Solver coordinates: Width Y, Depth X)
            ex_raw, ey_raw, ezj_raw = efields(neff, hx, hy, hzj, wavelength, cfg.dx, cfg.dy, epsxx=eps_x, epsyy=eps_y, epszz=eps_z)

            # Calculate Normalization Factor
            _, _, sz_field = poynting(ex_raw, ey_raw, ezj_raw, hx, hy, hzj)
            power = np.sum(sz_field) * cfg.dx * cfg.dy
            norm_factor = np.sqrt(376.73 / power)

            # Apply Normalization
            ex_norm = ex_raw * norm_factor
            ey_norm = ey_raw * norm_factor
            # Convert ezj (j*Ez) to Ez => Ez = ezj / 1j = -j * ezj
            ez_norm = (ezj_raw / 1j) * norm_factor

            # Transpose ALL fields back to (Depth X, Width Y)
            return ModeData(
                wavelength=wavelength,
                neff=neff,
                mx=1,
                my=1,
                ex=ex_norm.T,
                ey=ey_norm.T,
                ez=ez_norm.T,
                eps=eps_x.T,
                hx=hx.T,
                hy=hy.T,
                hz=hzj.T,
            )

    print("No TM mode found.")
    return None


def main() -> None:
    cfg = SimConfig()

    print("Calculating index profiles...")
    _, _, n_fw, n_shw = calculate_profiles(cfg)

    mat_fw = prepare_materials(n_fw)
    mat_shw = prepare_materials(n_shw)

    # Solve FW TM00
    mode_fw = solve_for_tm00(cfg, mat_fw, cfg.lambda_fw)
    if mode_fw:
        plot_field(mode_fw, "out/tm00_fw.html")
        save_mode_data(mode_fw, "E_fields_fw.mat")
    else:
        print("Failed to find FW TM00 mode.")

    # Solve SHW TM00
    mode_shw = solve_for_tm00(cfg, mat_shw, cfg.lambda_shw)
    if mode_shw:
        plot_field(mode_shw, "out/tm00_shw.html")
        save_mode_data(mode_shw, "E_fields_shw.mat")
    else:
        print("Failed to find SHW TM00 mode.")


if __name__ == "__main__":
    main()
