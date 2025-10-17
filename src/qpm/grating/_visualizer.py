from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import plotly.express as px  # pyright: ignore[reportMissingTypeStubs]
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
from plotly.subplots import make_subplots  # pyright: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    from ._builder import Grating


def _plot_structure_interactive(positions: jnp.ndarray, lengths: jnp.ndarray, kappas: jnp.ndarray) -> go.Figure:
    """
    Plots the physical structure of the grating interactively using Plotly.

    Args:
        positions: The starting position of each domain.
        lengths: The length of each domain.
        kappas: The coupling coefficient (κ) of each domain.

    Returns:
        A Plotly Figure object for interactive visualization.
    """
    hover_texts = [
        f"Index: {i}<br>Position: {p:.3f} μm<br>Length: {length:.3f} μm<br>Kappa: {k:.2f}"
        for i, (p, length, k) in enumerate(zip(np.array(positions), np.array(lengths), np.array(kappas), strict=False))
    ]

    fig = go.Figure(
        go.Bar(
            x=positions,
            y=kappas,
            width=lengths,
            marker_color=np.where(kappas > 0, "red", "blue"),
            hoverinfo="text",
            hovertext=hover_texts,
            name="Domains",
        ),
    )

    fig.update_layout(
        title_text="Interactive QPM Grating Structure",
        xaxis_title="Position (μm)",
        yaxis_title="Coupling Coefficient κ",
        bargap=0,
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 12},
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", gridcolor="lightgrey", tickformat=".1f")
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        gridcolor="lightgrey",
        zeroline=True,
        zerolinewidth=1.5,
        zerolinecolor="black",
    )
    return fig


def _plot_structure_heatmap(kappas: jnp.ndarray) -> go.Figure:
    """
    Visualizes the grating structure as a heatmap (barcode view) using Plotly.

    Args:
        kappas: The coupling coefficient (κ) of each domain.

    Returns:
        A Plotly Figure object for heatmap visualization.
    """
    kappa_img = np.array(kappas)[np.newaxis, :]

    fig = px.imshow(
        kappa_img,
        color_continuous_scale=["blue", "white", "red"],
        color_continuous_midpoint=0,
        aspect="auto",
        labels={"x": "Domain Index", "color": "Kappa"},
        title="QPM Grating Structure (Heatmap View)",
    )
    fig.update_yaxes(showticklabels=False, title="")
    return fig


def _plot_spectrum(positions: jnp.ndarray, kappas: jnp.ndarray, total_length: float) -> go.Figure:
    """
    Plots the spatial frequency spectrum interactively using Plotly.

    Args:
        positions: The starting position of each domain.
        kappas: The coupling coefficient (κ) of each domain.
        total_length: The total physical length of the grating.

    Returns:
        A Plotly Figure object for the spectrum visualization.
    """
    # Create a continuous representation for FFT
    n_samples = 8192
    z_samples = jnp.linspace(0, total_length, n_samples)
    indices = jnp.clip(jnp.searchsorted(positions, z_samples, side="right") - 1, 0)
    kappa_samples = kappas[indices]

    # Compute the FFT
    fft_vals = jnp.fft.fft(kappa_samples)
    fft_freqs = jnp.fft.fftfreq(n_samples, d=(total_length / n_samples))
    positive_mask = fft_freqs > 0

    # Create the plot
    fig = go.Figure(
        go.Scatter(
            x=fft_freqs[positive_mask],
            y=jnp.abs(fft_vals[positive_mask]),
            mode="lines",
            line={"width": 1.5},
            name="Magnitude",
        ),
    )

    fig.update_layout(
        title_text="Spatial Frequency Spectrum",
        xaxis_title="Spatial Frequency (μm⁻¹)",
        yaxis_title="Magnitude (a.u.)",
        yaxis_type="log",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 12},
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", gridcolor="lightgrey")
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        gridcolor="lightgrey",
        gridwidth=0.5,
        minor_gridcolor="lightgrey",
        minor_gridwidth=0.25,
    )
    return fig


def _plot_histogram(lengths: jnp.ndarray, kappas: jnp.ndarray) -> go.Figure:
    """
    Plots histograms of domain lengths and kappa values interactively using Plotly.

    Args:
        lengths: The length of each domain.
        kappas: The coupling coefficient (κ) of each domain.

    Returns:
        A Plotly Figure object containing the histograms.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Domain Length Distribution", "Coupling Coefficient Distribution"),
    )

    fig.add_trace(
        go.Histogram(x=np.array(lengths), name="Lengths", marker_color="skyblue", hovertemplate="Length: %{x}<br>Count: %{y}"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=np.array(kappas), name="Kappas", marker_color="lightcoral", hovertemplate="Kappa: %{x}<br>Count: %{y}"),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Domain Length (μm)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Coupling Coefficient κ", row=1, col=2)

    fig.update_layout(
        title_text=f"Distribution Analysis ({len(lengths):,} domains)",
        showlegend=False,
        plot_bgcolor="white",
        bargap=0.1,
        font={"family": "Arial, sans-serif", "size": 12},
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", gridcolor="lightgrey")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", gridcolor="lightgrey")
    return fig


def visualize(grating: "Grating", *, mode: str = "interactive") -> None:
    """
    Convenience function to visualize a Grating in various modes using Plotly.

    Args:
        grating: The Grating to visualize (N x 2 JAX array).
        mode: Visualization mode ('interactive', 'heatmap', 'spectrum', 'histogram').
    """
    if grating.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="Empty Grating", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, plot_bgcolor="white")
        fig.show()
        return

    # Pre-calculate common values
    lengths, kappas = grating[:, 0], grating[:, 1]
    positions = jnp.cumsum(jnp.concatenate([jnp.array([0.0]), lengths[:-1]]))
    total_length = float(jnp.sum(lengths))

    if mode == "interactive":
        fig = _plot_structure_interactive(positions, lengths, kappas)
    elif mode == "heatmap":
        fig = _plot_structure_heatmap(kappas)
    elif mode == "histogram":
        fig = _plot_histogram(lengths, kappas)
    elif mode == "spectrum":
        fig = _plot_spectrum(positions, kappas, total_length)
    else:
        available = "'interactive', 'heatmap', 'spectrum', 'histogram'"
        msg = f"Unknown mode: '{mode}'. Available modes are {available}."
        raise ValueError(msg)

    fig.show()
