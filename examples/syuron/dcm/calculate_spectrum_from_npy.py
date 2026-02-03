import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from qpm import cwes, mgoslt

jax.config.update("jax_enable_x64", val=True)


def load_and_simulate(file_path):
    print(f"Loading {file_path}...")
    try:
        data = np.load(file_path, allow_pickle=True)
        # Handle 0-d array containing dict
        if data.shape == ():
            data_dict = data.item()
        else:
            data_dict = data

        widths = jnp.array(data_dict["widths"])
        kappas = jnp.array(data_dict["kappas"])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

    # Simulation Config
    design_temp = 70.0
    wl_start = 1.0638
    wl_end = 1.0642
    wl_points = 1000

    wls = jnp.linspace(wl_start, wl_end, wl_points)
    dks = mgoslt.calc_twm_delta_k(wls, wls, design_temp)
    b_initial = jnp.array(1.0 + 0.0j)

    # JIT compile
    batch_simulate = jax.jit(jax.vmap(cwes.simulate_shg_npda, in_axes=(None, None, 0, None)))

    amps_out = batch_simulate(widths, kappas, dks, b_initial)
    amps_out.block_until_ready()

    spectrum = jnp.abs(amps_out)
    return wls, spectrum


def main():
    # Files
    file_std = "/workspaces/mictlan/domain_structure_Lc_std.npy"
    file_dith = "/workspaces/mictlan/domain_structure_Lc_dith.npy"

    print("Simulating Standard...")
    wls, spec_std = load_and_simulate(file_std)
    peak_std = jnp.max(spec_std)
    print(f"Standard Peak: {peak_std:.4e}")

    print("Simulating Dithered...")
    wls, spec_dith = load_and_simulate(file_dith)
    peak_dith = jnp.max(spec_dith)
    print(f"Dithered Peak: {peak_dith:.4e}")

    # Plotting
    print("Plotting results...")

    plt.figure(figsize=(10, 6))
    plt.plot(wls, spec_std, label=f"Standard (Peak: {peak_std:.2e})", color="red")
    plt.plot(wls, spec_dith, label=f"Dithered (Peak: {peak_dith:.2e})", color="green")

    plt.title(f"Comparison of Quantization Methods at Lc Resolution")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()
    plt.grid(True)

    # Save to PNG
    out_img = "spectrum_comparison.png"
    plt.savefig(out_img, dpi=300)
    print(f"Plot saved to {out_img}")


if __name__ == "__main__":
    main()
