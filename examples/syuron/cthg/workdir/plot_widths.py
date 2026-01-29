import argparse
import pickle
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from qpm import cwes2, mgoslt

# Physics Constants
WAVELENGTH = 1.064
TEMPERATURE = 70.0


@dataclass(frozen=True)
class SimulationData:
    """Container for domain widths and metadata."""

    widths: np.ndarray
    filename: str

    @property
    def total_length_mm(self) -> float:
        return float(np.sum(self.widths)) / 1000


def load_data(filename: str) -> SimulationData:
    """Loads widths from a pickle file."""
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return SimulationData(widths=np.abs(data["widths"]), filename=filename)


def compute_precise_amplitude(widths: np.ndarray) -> float:
    """Performs high-precision amplitude re-verification using JAX."""
    dk1 = mgoslt.calc_twm_delta_k(WAVELENGTH, WAVELENGTH, TEMPERATURE)
    dk2 = mgoslt.calc_twm_delta_k(WAVELENGTH, WAVELENGTH / 2, TEMPERATURE)

    # Scaling factors derived from the physical model
    k0 = 1.5e-5 / (2 / np.pi)
    b_init = jnp.array([jnp.sqrt(10.0), 0.0, 0.0], dtype=jnp.complex128)

    n = len(widths)
    signs = jnp.tile(jnp.array([1.0, -1.0]), (n // 2 + 1))[:n]

    b_final = cwes2.simulate_twm(jnp.array(widths), signs * k0, signs * k0 * 2, dk1, dk2, b_init)
    return float(jnp.abs(b_final[2]))


def save_plot(data: SimulationData, amp: float, scale: float = 1.0) -> None:
    """Generates and saves a publication-quality distribution plot."""
    # Scale affects fonts and markers, but we keep figure size reasonable
    # to ensure the relative size of text increases.
    fs = 10 * scale
    ms = 0.5 * scale

    plt.figure(figsize=(10, 6), dpi=150)
    plt.scatter(range(len(data.widths)), data.widths, s=ms, alpha=0.6, color="#1f77b4")

    # Titles and Labels
    plt.suptitle(f"Domain Width Distribution: {data.filename}", fontsize=fs * 1.2)
    plt.title(f"Amp: {amp:.5f} | Length: {data.total_length_mm:.3f} mm", fontsize=fs)

    plt.xlabel("Domain Index", fontsize=fs)
    plt.ylabel("Width (μm)", fontsize=fs)
    plt.xticks(fontsize=fs * 0.8)
    plt.yticks(fontsize=fs * 0.8)

    plt.grid(visible=True, linestyle="--", alpha=0.3)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    output = data.filename.replace(".pkl", "_width_dist.png")
    plt.savefig(output, bbox_inches="tight")
    print(f"Successfully saved plot to {output} (scale={scale})")


def main() -> None:
    jax.config.update("jax_enable_x64", val=True)

    parser = argparse.ArgumentParser(description="QPM domain width distribution plotter")
    parser.add_argument("filename", help="Path to the simulation .pkl file")
    parser.add_argument("--scale", type=float, default=1.2, help="Scale factor for plot elements (default: 1.2)")
    args = parser.parse_args()

    try:
        data = load_data(args.filename)
        amp = compute_precise_amplitude(data.widths)
        print(f"Loaded {len(data.widths)} domains from {data.filename}")
        print(f"Precise Amplitude: {amp:.6f}")
        save_plot(data, amp, scale=args.scale)
    except (FileNotFoundError, pickle.UnpicklingError, KeyError) as e:
        print(f"Error processing file: {e}")
    except RuntimeError as e:
        print(f"Calculation error: {e}")


if __name__ == "__main__":
    main()
