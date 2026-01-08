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
    ey: np.ndarray
    eps_y: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    hz: np.ndarray


def prepare_materials(n_dist: np.ndarray) -> dict[str, np.ndarray]:
    """Prepares material dictionary from refractive index distribution."""
    # Solver expects (Y, X) for epsxx etc.
    # calculate_profiles returns (X, Y).
    # so we define eps as n^2 and transpose it.
    eps = n_dist**2
    return {
        "EpsX": eps.T,
        "EpsY": eps.T,
        "EpsZ": eps.T,
    }


def _get_polarization(hx: np.ndarray, hy: np.ndarray) -> int:
    """Determines polarization: 2 for TM, 1 for TE."""
    mag_hx = np.sum(np.abs(hx) ** 2)
    mag_hy = np.sum(np.abs(hy) ** 2)
    return 2 if mag_hx > mag_hy else 1


def plot_field(mode: ModeData, out_path: str) -> None:
    """Plots and saves a heatmap of the electric field."""
    fig = go.Figure(data=go.Heatmap(z=np.abs(mode.ey), colorscale="Viridis", colorbar={"title": "Electric Field |Ey|"}))
    fig.update_layout(
        title=f"TM{mode.mx - 1}{mode.my - 1} Mode Field Distribution (wl={mode.wavelength} um)",
        xaxis_title="Width y (approx)",
        yaxis_title="Depth x (approx)",
    )
    fig.write_html(out_path)
    print(f"Saved {out_path}")


def save_mode_data(mode: ModeData, filename: str) -> None:
    """Saves mode data to a .mat file."""
    data = {
        "wavelength": mode.wavelength,
        "neff": mode.neff,
        "mx": mode.mx,
        "my": mode.my,
        "ey": mode.ey,
        "eps_y": mode.eps_y,
        "hx": mode.hx,
        "hy": mode.hy,
        "hz": mode.hz,
    }
    scipy.io.savemat(filename, data)
    print(f"Saved mode data to {filename}")


def solve_for_tm00(
    cfg: SimConfig,
    materials: dict[str, np.ndarray],
    wavelength: float,
    n_modes: int = 1,
) -> ModeData | None:
    print(f"Solving for wl={wavelength:.3f} um, searching for fundamental TM mode...")

    eps_x = materials["EpsX"]
    eps_y = materials["EpsY"]
    eps_z = materials["EpsZ"]

    neffs, hxs, hys, hzjs = modesolver.wgmodes(
        wavelength,
        2.2,
        n_modes,
        cfg.dx,
        cfg.dy,
        "0000",
        epsxx=eps_x,
        epsyy=eps_y,
        epszz=eps_z,
    )

    if hxs.ndim == 2:
        hxs = hxs[:, :, np.newaxis]
        hys = hys[:, :, np.newaxis]
        hzjs = hzjs[:, :, np.newaxis]
        neffs = np.atleast_1d(neffs)

    for m in range(n_modes):
        hx = hxs[:, :, m]
        hy = hys[:, :, m]
        neff = neffs[m]

        pol = _get_polarization(hx, hy)

        if pol == 2:
            print(f"Found Fundamental TM Mode (TM00): neff={neff:.4f} (Mode index {m})")

            hzj = hzjs[:, :, m]
            ex, ey, ezj = efields(neff, hx, hy, hzj, wavelength, cfg.dx, cfg.dy, epsxx=eps_x, epsyy=eps_y, epszz=eps_z)
            _, _, sz_field = poynting(ex, ey, ezj, hx, hy, hzj)

            power = np.sum(sz_field) * cfg.dx * cfg.dy
            norm_factor = np.sqrt(376.73 / power)
            ey_norm = ey * norm_factor

            return ModeData(
                wavelength=wavelength,
                neff=neff,
                mx=1,
                my=1,
                ey=ey_norm,
                eps_y=eps_y,
                hx=hx,
                hy=hy,
                hz=hzj,
            )

    print("No TM mode found.")
    return None


def main() -> None:
    cfg = SimConfig()

    # Get Materials directly
    print("Calculating index profiles...")
    _, _, n_fw, n_shw = calculate_profiles(cfg)

    mat_fw = prepare_materials(n_fw)
    mat_shw = prepare_materials(n_shw)

    # Solve FW TM00
    mode_fw = solve_for_tm00(cfg, mat_fw, cfg.lambda_fw)
    if mode_fw:
        plot_field(mode_fw, "out/tm00_fw.html")
        save_mode_data(mode_fw, "E_fields_fw_veryhigh_res.mat")
    else:
        print("Failed to find FW TM00 mode.")

    # Solve SHW TM00
    mode_shw = solve_for_tm00(cfg, mat_shw, cfg.lambda_shw)
    if mode_shw:
        plot_field(mode_shw, "out/tm00_shw.html")
        save_mode_data(mode_shw, "E_fields_shw_veryhigh_res.mat")
    else:
        print("Failed to find SHW TM00 mode.")


if __name__ == "__main__":
    main()
