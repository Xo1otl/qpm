from dataclasses import dataclass

import jax
import jax.numpy as jnp
from plotly import graph_objects as go

from qpm import cwes2, mgoslt

# Constants
NORO_CRR_FACTOR = 1.07 / 2.84 * 100
DESIGN_TEMP = 70.0
KAPPA_MAG = 1.31e-5 / (2 / jnp.pi)
NUM_PERIODS = 600
DESIGN_WL = 1.031
WL_RANGE = jnp.linspace(1.025, 1.037, 1000)


@dataclass(frozen=True)
class DomainConfig:
    name: str
    widths: jax.Array
    signs: jax.Array


def main() -> None:
    jax.config.update("jax_enable_x64", val=True)

    # 1. Physics Setup
    dk_val = mgoslt.calc_twm_delta_k(DESIGN_WL, DESIGN_WL, DESIGN_TEMP)
    Lp = 2 * jnp.pi / dk_val
    dks = mgoslt.calc_twm_delta_k(WL_RANGE, WL_RANGE, DESIGN_TEMP)

    # 2. Geometry Definitions (Fixed Duty D=0.5)
    # 2-Region (Left-Anchored): [Pulse(+), Gap(-)]
    # 3-Region (Center-Anchored): [Gap(-), Pulse(+), Gap(-)]
    configs = [
        DomainConfig(
            name="2-Region (Left-Anchored)",
            widths=jnp.array([0.5, 0.5]) * Lp,
            signs=jnp.array([1.0, -1.0]),
        ),
        DomainConfig(
            name="3-Region (Center-Anchored)",
            widths=jnp.array([0.25, 0.5, 0.25]) * Lp,
            signs=jnp.array([-1.0, 1.0, -1.0]),
        ),
    ]

    # 3. Simulation & Execution
    simulate = jax.jit(jax.vmap(cwes2.simulate_shg_npda, in_axes=(None, None, 0, None)))
    b0 = jnp.array(1.0 + 0.0j)

    fig = go.Figure()

    for cfg in configs:
        print(f"Simulating {cfg.name}...")
        widths = jnp.tile(cfg.widths, NUM_PERIODS)
        kappas = jnp.tile(cfg.signs * KAPPA_MAG, NUM_PERIODS)

        amps = simulate(widths, kappas, dks, b0)
        effs = jnp.abs(amps) ** 2 * NORO_CRR_FACTOR

        fig.add_trace(go.Scatter(x=WL_RANGE, y=effs, mode="lines", name=cfg.name))

    # 4. Visualization
    fig.update_layout(
        title="SHG Efficiency Comparison: 2-Region vs 3-Region (D=0.5)",
        xaxis_title="Wavelength (µm)",
        yaxis_title="Efficiency",
        height=600,
        width=900,
    )
    fig.show()


if __name__ == "__main__":
    main()
