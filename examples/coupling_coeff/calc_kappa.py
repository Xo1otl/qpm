from dataclasses import dataclass

import numpy as np
import scipy.io
from calc_index import SimConfig


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


def load_mode_data(filename: str) -> ModeData:
    """Loads mode data from a .mat file."""
    data = scipy.io.loadmat(filename)

    return ModeData(
        wavelength=data["wavelength"],
        neff=data["neff"],
        mx=data["mx"],
        my=data["my"],
        ey=data["ey"],
        eps_y=data["eps_y"],
        hx=data["hx"],
        hy=data["hy"],
        hz=data["hz"],
    )


def calculate_overlap(cfg: SimConfig, mode_fw: ModeData, mode_shw: ModeData, e2yy: float) -> float:
    """Calculates the overlap integral (coupling coefficient)."""
    ey_fw = mode_fw.ey
    ey_shw = mode_shw.ey
    eps_shw = mode_shw.eps_y

    d33 = 1.38e-5
    e0 = 8.854e-18
    c = 2.99792458e14

    core_mask = np.isclose(eps_shw, e2yy, atol=1e-4)
    term = core_mask * (ey_fw**2) * ey_shw
    sz = d33 * np.real(term)

    da = cfg.dx * cfg.dy
    ol_lin = np.sum(sz) * da

    return np.abs((4 * np.pi * c / cfg.lambda_fw) * e0 * ol_lin / 4) * 1e4


def main() -> None:
    cfg = SimConfig()

    try:
        print("Loading mode data...")
        mode_fw = load_mode_data("E_fields_fw.mat")
        mode_shw = load_mode_data("E_fields_shw.mat")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Make sure E_fields_fw.mat and E_fields_shw.mat exist in the current directory.")
        return

    # e2yy is max permittivity of SHW mode (from plot_efields logic)
    e2yy = np.max(mode_shw.eps_y)

    print(f"Calculated using dx={cfg.dx}, dy={cfg.dy}, lambda_fw={cfg.lambda_fw}...")
    kappa = calculate_overlap(cfg, mode_fw, mode_shw, e2yy)
    print(rf"Overlap Integral (kappa): {kappa:.4f} $\text{{W}}^{-1 / 2} \text{{cm}}^{-1}$")


if __name__ == "__main__":
    main()
