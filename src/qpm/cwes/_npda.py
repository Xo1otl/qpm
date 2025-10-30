# TODO: Implement the code to calculate the SHG and THG spectrum under the NPDA in a single step using a Fourier transform.
# TODO: THG efficiency requires a double Fourier transform.
# TODO: The function names should be ft_shg_spectrum and ft_thg_spectrum.
# TODO: Rename calculate_phys_eff_spectrum to calc_phys_spectrum.
# TODO" The squaring operation should be moved to the caller, as there might be cases where the complex amplitude is needed.
'''
from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
import plotly.graph_objects as go

from qpm import grating, mgoslt


@dataclass(frozen=True)
class DeviceConfig:
    length: float
    num_domains: int
    kappa: float


@dataclass(frozen=True)
class FourierConfig:
    k_center: float
    k_bandwidth: float
    num_points: int


def new_fourier_config(
    dk_eff: jax.Array,
    device: DeviceConfig,
    num_points: int,
    bw_scale: float = 1.2,
) -> "FourierConfig":
    k_norm_center = device.num_domains / 2.0
    dk_eff_range = jnp.max(dk_eff) - jnp.min(dk_eff)
    k_norm_bw = dk_eff_range * device.length / (2 * jnp.pi) * bw_scale
    return FourierConfig(
        k_center=k_norm_center,
        k_bandwidth=float(k_norm_bw),
        num_points=num_points,
    )


@partial(jax.jit, static_argnames=("m", "fft_len"))
def czt(x: jax.Array, m: int, fft_len: int, w: jax.Array, a: jax.Array) -> jax.Array:
    """Computes the Chirp Z-Transform of a signal."""
    n = x.shape[-1]
    n_range = jnp.arange(n)
    y = x * (a**-n_range) * w ** (n_range**2 / 2)
    k_range_full = jnp.arange(-(n - 1), m)
    h = w ** (-(k_range_full**2) / 2)
    y_fft = jnp.fft.fft(y, n=fft_len)
    h_fft = jnp.fft.fft(h, n=fft_len)
    conv_result = jnp.fft.ifft(y_fft * h_fft)
    k_range_out = jnp.arange(m)
    final_chirp = w ** (k_range_out**2 / 2)
    return conv_result[n - 1 : n - 1 + m] * final_chirp


def create_alternating_signal_from_widths(widths: jax.Array, n_points: int = 8192) -> jax.Array:
    """
    Generates an alternating +1/-1 signal based on an array of domain widths.
    """
    num_domains = widths.shape[0]
    alternating_values = jnp.power(-1, jnp.arange(num_domains))
    cumulative_widths = jnp.cumsum(widths)
    total_width = cumulative_widths[-1]
    boundary_points = jnp.round(cumulative_widths / total_width * n_points).astype(jnp.int32)
    boundary_points = jnp.insert(boundary_points, 0, 0)
    boundary_points = boundary_points.at[-1].set(n_points)
    points_per_domain = jnp.diff(boundary_points)
    return jnp.repeat(alternating_values, points_per_domain)


def calculate_fourier_spectrum(
    signal: jax.Array,
    config: FourierConfig,
) -> tuple[jax.Array, jax.Array]:
    """Calculates the spectrum of a signal using CZT for a specified frequency window."""
    fs = signal.shape[0]
    k_start = config.k_center - config.k_bandwidth / 2.0
    k_end = config.k_center + config.k_bandwidth / 2.0
    f_norm_start = k_start / fs
    f_norm_end = k_end / fs
    w = jnp.exp(-1j * 2 * jnp.pi * (f_norm_end - f_norm_start) / config.num_points)
    a = jnp.exp(1j * 2 * jnp.pi * f_norm_start)
    required_len = fs + config.num_points - 1
    fft_len = 1 << (required_len - 1).bit_length()
    spectrum_raw = czt(signal, m=config.num_points, fft_len=fft_len, w=w, a=a)
    spectrum_amp_normalized = jnp.abs(spectrum_raw * 2 * jnp.pi / fs)
    k_axis_normalized = jnp.linspace(k_start, k_end, config.num_points)
    return k_axis_normalized, spectrum_amp_normalized


@dataclass(frozen=True)
class ConvParams:
    norm_k_axis: jax.Array
    k_center: float
    length: float
    kappa: float
    dk_eff: jax.Array


def calculate_phys_eff_spectrum(norm_spectrum_amp: jax.Array, params: ConvParams) -> jax.Array:
    """Converts the normalized Fourier spectrum to physical units."""
    k_deviation_norm = params.norm_k_axis - params.k_center
    dk_total_phys = k_deviation_norm * (2 * jnp.pi / params.length)
    # Convert the normalized Fourier transform (|F_norm|) to the physical one (|F_phys|)
    # |F_phys| = (L / 2π) * |F_norm|
    fourier_amp_phys = norm_spectrum_amp * (params.length / (2 * jnp.pi))
    # Efficiency is proportional to |κ_d * F_phys(Δk)|^2
    eff = (params.kappa**2) * (fourier_amp_phys**2)
    return jnp.interp(params.dk_eff, dk_total_phys, eff)


def plot_spectrum(wls: jax.Array, effs: jax.Array) -> None:
    """Plots the SHG efficiency spectrum using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wls, y=effs, mode="lines", name="Fourier Method"))
    fig.update_layout(
        title="SHG Spectrum (Fourier Method)",
        xaxis_title="Fundamental Wavelength (μm)",
        yaxis_title="Normalized SHG Efficiency",
        template="plotly_white",
    )
    fig.show()


def calc_dk_eff(dk_material: jax.Array, device: DeviceConfig) -> jax.Array:
    """Calculates the effective phase mismatch."""
    k_norm_center = device.num_domains / 2.0
    k_g = k_norm_center * (2 * jnp.pi / device.length)
    return dk_material - k_g


def calc_shg_effs(widths: jax.Array, wls: jax.Array, device: DeviceConfig, design_temp: float) -> jax.Array:
    dk_material = mgoslt.calc_twm_delta_k(wls, wls, design_temp)
    dk_eff = calc_dk_eff(dk_material, device)
    signal = create_alternating_signal_from_widths(widths, 30000)
    fourier_config = new_fourier_config(dk_eff, device, wls.shape[0])

    k_ax_norm, spec_amp_norm = calculate_fourier_spectrum(signal, fourier_config)
    conv_params = ConvParams(
        norm_k_axis=k_ax_norm,
        k_center=fourier_config.k_center,
        length=device.length,
        kappa=device.kappa,
        dk_eff=dk_eff,
    )
    return calculate_phys_eff_spectrum(spec_amp_norm, conv_params)
'''
