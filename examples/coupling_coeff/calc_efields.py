from dataclasses import dataclass

import modesolver
import numpy as np
import plotly.graph_objects as go
from calc_index import SimConfig, calculate_profiles
from modesolver.postprocess import efields
from modesolver.postprocess.poynting import poynting
from scipy.signal import find_peaks


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


def solve_for_tm00(
    cfg: SimConfig,
    materials: dict[str, np.ndarray],
    wavelength: float,
    n_modes: int,
) -> ModeData | None:
    """Runs modesolver and searches for TM00 mode (1,1)."""
    print(f"Solving for wl={wavelength:.3f} um, TM00...")

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

    # Handle case where n_modes=1 and solver returns 2D arrays
    if hxs.ndim == 2:
        hxs = hxs[:, :, np.newaxis]
        hys = hys[:, :, np.newaxis]
        hzjs = hzjs[:, :, np.newaxis]
        neffs = np.atleast_1d(neffs)

    for m in range(n_modes):
        hx = hxs[:, :, m]
        hy = hys[:, :, m]
        hzj = hzjs[:, :, m]
        neff = neffs[m]

        if _get_polarization(hx, hy) == 2:  # TM
            mx, my = _estimate_mode_indices(hx)

            # Target TM00 corresponds to indices (1,1) in this convention
            if mx == 1 and my == 1:
                print(f"Found TM00 Mode (1,1): neff={neff:.4f}")

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


def main() -> None:
    cfg = SimConfig()
    n_modes = 1

    # Get Materials directly
    print("Calculating index profiles...")
    _, _, n_fw, n_shw = calculate_profiles(cfg)

    mat_fw = prepare_materials(n_fw)
    mat_shw = prepare_materials(n_shw)

    # Solve FW TM00
    mode_fw = solve_for_tm00(cfg, mat_fw, cfg.lambda_fw, n_modes)
    if mode_fw:
        plot_field(mode_fw, "out/tm00_fw.html")
    else:
        print("Failed to find FW TM00 mode.")

    # Solve SHW TM00
    mode_shw = solve_for_tm00(cfg, mat_shw, cfg.lambda_shw, n_modes)
    if mode_shw:
        plot_field(mode_shw, "out/tm00_shw.html")
    else:
        print("Failed to find SHW TM00 mode.")


if __name__ == "__main__":
    main()
