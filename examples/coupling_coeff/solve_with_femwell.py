import os

# Force JAX to use CPU to avoid GPU contention (since FEM is CPU-bound anyway)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import matplotlib.pyplot as plt
import numpy as np

from qpm import config, wgmode


def verify_single_mode_condition(results: wgmode.ModeList) -> None:
    guided_modes = [m for m in results if m.is_guided]
    guided_count = len(guided_modes)

    print(f"Verification: Found {guided_count} guided modes.")

    if guided_count == 0:
        print("FAIL: No modes guided.")
    elif guided_count == 1:
        print("SUCCESS: Single-Mode operation achieved (Fundamental only).")
    else:
        print(f"FAIL: Multi-mode behavior detected ({guided_count} modes guided).")
        for m in guided_modes:
            print(f"  - Mode {m.index}: n_eff={m.n_eff:.6f}")


def plot_mode_result(ctx: wgmode.SimulationContext, result: wgmode.ModeResult) -> None:
    if not result.is_guided:
        print(f"Mode {result.index}: n_eff = {result.n_eff:.6f} (Cutoff - Ignored)")
        return

    print(f"Mode {result.index}: n_eff = {result.n_eff:.6f} (Guided)")

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot A: Refractive Index ---
    n_core_max = np.max(ctx.n_dist)
    n_core_min = np.min(ctx.n_dist)
    print(f"Refractive Index Range: min={n_core_min:.5f}, max={n_core_max:.5f}")

    ctx.basis.plot(
        ctx.n_dist,
        ax=axes[0],
        cmap="viridis",
        shading="gouraud",
        # Use data limits to maximize contrast
        vmin=n_core_min,
        vmax=n_core_max,
    )

    mappable = axes[0].collections[-1]
    plt.colorbar(mappable, ax=axes[0], label="n")
    axes[0].set_title(f"Refractive Index (Clamped to {ctx.n_sub:.3f}+)")
    axes[0].set_aspect("equal")

    # --- Plot B: Mode Intensity ---
    result.field_data.plot_intensity(ax=axes[1], colorbar=True)
    axes[1].set_title(f"Mode {result.index} Intensity (n_eff={result.n_eff:.5f})")
    axes[1].set_aspect("equal")
    axes[1].set_ylabel("")

    plt.tight_layout()
    filename = f"out/sim_mode_{result.index}.png"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()


def plot_mode_manual(result: wgmode.ModeResult) -> None:
    """
    Manually plots the mode intensity using matplotlib.tripcolor.
    Reflects the internal field data structure of femwell.
    """
    print(f"Manual Plotting for Mode {result.index}...")

    # 1. Access Mode and Mesh
    mode = result.field_data
    basis = mode.basis
    mesh = basis.mesh

    # 2. Extract Coordinates and Triangles
    # mesh.p is [2, N_points] -> x, y
    x = mesh.p[0, :]
    y = mesh.p[1, :]
    # mesh.t is [3, N_triangles] -> indices
    triangles = mesh.t.T

    # 3. Extract Field
    E = mode.E
    n_vertices = x.shape[0]

    # Handle Higher-Order Elements (e.g. P2 has more DOFs than vertices)
    # For tripcolor, we typically just want the vertex values.
    if E.shape[0] > n_vertices:
        # Assuming typical ordering where vertices come first
        E_vertex = E[:n_vertices]
    elif E.shape[0] == n_vertices:
        E_vertex = E
    elif E.shape[0] == 3 * basis.doflocs.shape[1]:
        # Vector mode case handled for completeness, though TM00 is usually scalar-ish in this solver context
        # But wait, wgmode usually returns scalar mode approximations or specific components?
        # Let's stick to the simple check we verified in the test script.
        # In the test script: "Field matches doflocs count" was true for 212101 DOFs vs 53226 mesh points.
        # So it is higher order. Converting to intensity first might be safer if we map correctly?
        # Actually my test script confirmed:
        # Mesh points (N_p): (53226,)
        # Basis doflocs shape: (2, 212101)
        # So we HAVE TO truncate/subsample to N_p for tripcolor on `triangles` (which references N_p vertices).
        E_vertex = E[:n_vertices]
    else:
        # Fallback
        E_vertex = E[:n_vertices]

    # 4. Calculate Intensity
    intensity = np.abs(E_vertex) ** 2

    # 5. Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Tripcolor with Gouraud shading interpolates colors between vertices
    img = ax.tripcolor(x, y, triangles, intensity, shading="gouraud")

    plt.colorbar(img, ax=ax, label="Intensity |E|^2")
    ax.set_aspect("equal")
    ax.set_title(f"Mode {result.index} Manual Plot (n_eff={result.n_eff:.5f})")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")

    plt.tight_layout()
    filename = f"out/sim_mode_{result.index}_manual.png"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()


def run() -> None:
    print("--- RUNNING WAVEGUIDE SIMULATION (TM00 ONLY) ---")

    # 1. Initialization
    pp = config.new_process_params()
    cfg = config.new_simulation_config(wavelength_um=1.064, process_params=pp)
    # We can reduce num_modes since we only want TM00, but compute_tm00 handles logic
    cfg.plot_modes = True

    # 2. Computation - Only TM00
    print("Calculating only TM00 mode...")
    tm00 = wgmode.compute_tm00(cfg)

    if tm00 is None:
        print("Error: TM00 mode not found!")
        return

    print(f"Substrate Index (n_sub): {cfg.n_sub:.6f}")

    # 3. Visualization
    # FEMWell's built-in plot
    ctx, _ = wgmode.compute_modes_from_config(cfg)  # Re-get context primarily for refractive index plot arg
    plot_mode_result(ctx, tm00)

    # Manual plot
    plot_mode_manual(tm00)

    # 4. Verification
    print(f"Verification: TM00 Mode Found. n_eff={tm00.n_eff:.6f}")


if __name__ == "__main__":
    run()
