import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from compute import SimulationResult, calculate_mse


def plot_results(
    magnus_res: SimulationResult,
    dop853_res: SimulationResult,
    time_magnus: float,
    time_dop853: float,
    filename: str = "amp_trace_comparison.html",
) -> None:
    # Calculate MSE
    # Note: calculate_mse handles interpolation if needed
    mse = calculate_mse(magnus_res, dop853_res)
    
    print(f"MSE (Magnus vs DOP853): FW={mse[0]:.2e}, SHW={mse[1]:.2e}, THW={mse[2]:.2e}")

    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            f"Fundamental Wave (1ω) | MSE={mse[0]:.1e}",
            f"Second Harmonic (2ω) | MSE={mse[1]:.1e}",
            f"Third Harmonic (3ω) | MSE={mse[2]:.1e}",
            "Total Power"
        )
    )

    # Common style for Magnus (Scatter markers)
    magnus_marker = dict(size=6, symbol='circle')
    # Common style for DOP853 (Line)
    dop853_line = dict(width=2, dash='solid')

    # 1. Fundamental Wave (A1)
    fig.add_trace(
        go.Scatter(
            x=magnus_res.z, y=np.abs(magnus_res.a1),
            mode='markers',
            name=f"Magnus ({time_magnus:.4f}s)",
            marker=magnus_marker,
            legendgroup="magnus",
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dop853_res.z, y=np.abs(dop853_res.a1),
            mode='lines',
            name=f"DOP853 ({time_dop853:.4f}s)",
            line=dop853_line,
            legendgroup="dop853",
        ),
        row=1, col=1
    )

    # 2. Second Harmonic (A2)
    fig.add_trace(
        go.Scatter(
            x=magnus_res.z, y=np.abs(magnus_res.a2),
            mode='markers',
            name="Magnus (2ω)",
            marker=magnus_marker,
            legendgroup="magnus",
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dop853_res.z, y=np.abs(dop853_res.a2),
            mode='lines',
            name="DOP853 (2ω)",
            line=dop853_line,
            legendgroup="dop853",
            showlegend=False
        ),
        row=2, col=1
    )

    # 3. Third Harmonic (A3)
    fig.add_trace(
        go.Scatter(
            x=magnus_res.z, y=np.abs(magnus_res.a3),
            mode='markers',
            name="Magnus (3ω)",
            marker=magnus_marker,
            legendgroup="magnus",
            showlegend=False
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dop853_res.z, y=np.abs(dop853_res.a3),
            mode='lines',
            name="DOP853 (3ω)",
            line=dop853_line,
            legendgroup="dop853",
            showlegend=False
        ),
        row=3, col=1
    )

    # 4. Total Power
    fig.add_trace(
        go.Scatter(
            x=magnus_res.z, y=magnus_res.total_power,
            mode='markers',
            name="Magnus Power",
            marker=magnus_marker,
            legendgroup="magnus",
            showlegend=False
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dop853_res.z, y=dop853_res.total_power,
            mode='lines',
            name="DOP853 Power",
            line=dop853_line,
            legendgroup="dop853",
            showlegend=False
        ),
        row=4, col=1
    )

    # Update layout
    fig.update_layout(
        height=1000,
        title_text="Amplitude Trace Accuracy Comparison",
        showlegend=True,
        template="plotly_white",
        font=dict(size=14)
    )

    fig.update_xaxes(title_text="Position (μm)", row=4, col=1)
    
    # Update y-axes labels
    fig.update_yaxes(title_text="|A1|", row=1, col=1)
    fig.update_yaxes(title_text="|A2|", row=2, col=1)
    fig.update_yaxes(title_text="|A3|", row=3, col=1)
    fig.update_yaxes(title_text="Power", row=4, col=1)

    print(f"Saving plot to {filename}...")
    fig.write_html(filename)
    print("Done.")
