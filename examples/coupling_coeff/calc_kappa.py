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


def calculate_overlap(cfg: SimConfig, mode_fw: ModeData, mode_shw: ModeData) -> float:
    """Calculates the overlap integral (coupling coefficient)."""
    ey_fw = mode_fw.ey
    ey_shw = mode_shw.ey

    # d33 value
    d33 = 13.8e-12 * 1e6  # 13.8 pm/V -> 13.8e-12 m/V -> 1.38e-5 um/V
    # Or keep consistent units: if everything is in microns...
    # c = 3e8 m/s = 3e14 um/s. e0 = 8.854e-12 F/m = 8.854e-18 F/um.
    # d33 = 13.8 pm/V = 13.8e-12 m/V = 1.38e-5 um/V.
    d33 = 1.38e-5

    # Effective nonlinearity d_eff = (2/pi) * d33
    d_eff = (2 / np.pi) * d33

    e0 = 8.854e-18
    c = 2.99792458e14
    omega_fw = 2 * np.pi * c / cfg.lambda_fw
    omega_shw = 2 * omega_fw  # 2*omega

    # Actually, usually overlap is over all space.
    # But if we want to restrict to where nonlinearity exists (the core/domain inverted region?)
    # The domain inversion profile is usually implicitly handled by the sign of d_eff in QPM.
    # Here assuming perfect first-order QPM where d_eff compensates phase.

    term = np.conj(ey_shw) * (ey_fw**2)
    # The overlap integral I
    da = cfg.dx * cfg.dy
    overlap_integral = np.sum(d_eff * term) * da

    # Kappa = (omega_fw * epsilon_0 / 4) * Integral
    # Wait, Reference: kappa_SHG = (omega_2 / 4) * ... ?
    # Usually coupled mode equations: dA2/dz = i * kappa * A1^2 e^...
    # Let's trust the formula in calc_process.priv.md:
    # kappa_SHG = (omega_2 * epsilon_0 / 4) * d_eff * Integral
    # omega_2 is SHW frequency?
    #   Text says: kappa_SHG = (omega_2 epsilon_0 / 4) ...
    #   Using omega_shw.

    kappa_val = (omega_shw * e0 / 4) * overlap_integral

    # Convert to cm^-1 ? Result is in um^-1 since everything is um.
    # 1 um^-1 = 1e4 cm^-1.
    return np.abs(kappa_val) * 1e4


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

    print(f"Calculated using dx={cfg.dx}, dy={cfg.dy}, lambda_fw={cfg.lambda_fw}...")
    kappa = calculate_overlap(cfg, mode_fw, mode_shw)
    print(rf"Overlap Integral (kappa): {kappa:.4f} $\text{{W}}^{-1 / 2} \text{{cm}}^{-1}$")


if __name__ == "__main__":
    main()
