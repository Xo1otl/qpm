import numpy as np
import matplotlib.pyplot as plt


def simulate_propagation(length, delta_k, num_steps=1000, qpm=False):
    z = np.linspace(0, length, num_steps)
    dz = z[1] - z[0]
    d_coeff = np.ones_like(z)

    if qpm and delta_k != 0:
        d_coeff = np.sign(np.cos(delta_k * z))

    integrand = d_coeff * np.exp(-1j * delta_k * z)
    E = np.cumsum(integrand) * dz
    return z, E


def plot_visualizations():
    L = 40
    dk_pm = 0.0
    z_pm, E_pm = simulate_propagation(L, dk_pm)
    dk_mm = 4 * np.pi / L
    z_mm, E_mm = simulate_propagation(L, dk_mm)
    dk_qpm = dk_mm
    z_qpm, E_qpm = simulate_propagation(L, dk_qpm, qpm=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1: Phasor Diagram ---
    ax_complex = axes[0]
    ax_complex.plot(np.real(E_pm), np.imag(E_pm), label="Phase Matching (PM)", color="C2", linewidth=2)
    ax_complex.plot(np.real(E_mm), np.imag(E_mm), label="Phase Mismatch", color="C1", linewidth=2, linestyle="--")
    ax_complex.plot(np.real(E_qpm), np.imag(E_qpm), label="Quasi-Phase Matching (QPM)", color="C0", linewidth=2)
    ax_complex.set_title("Phasor Diagram (Complex Plane)\nEvolution of Electric Field E(z)", fontsize=14)
    ax_complex.set_xlabel("Real part of E", fontsize=12)
    ax_complex.set_ylabel("Imaginary part of E", fontsize=12)
    ax_complex.axhline(0, color="gray", alpha=0.3)
    ax_complex.axvline(0, color="gray", alpha=0.3)
    ax_complex.grid(True, linestyle=":", alpha=0.6)
    ax_complex.legend(fontsize=10)
    ax_complex.set_aspect("equal")

    # Numbers removed for Subplot 1 (Conceptual)
    ax_complex.set_xticks([])
    ax_complex.set_yticks([])

    # --- Subplot 2: Amplitude Growth ---
    ax_amp = axes[1]
    amp_pm = np.abs(E_pm)
    amp_mm = np.abs(E_mm)
    amp_qpm = np.abs(E_qpm)

    ax_amp.plot(z_pm, amp_pm, label=r"Phase Matching ($\Delta k = 0$)", color="C2")
    ax_amp.plot(z_mm, amp_mm, label=r"Phase Mismatch ($\Delta k \neq 0$)", color="C1", linestyle="--")
    ax_amp.plot(z_qpm, amp_qpm, label="QPM", color="C0")

    Lc = np.pi / dk_qpm
    num_domains = int(L / Lc)
    for i in range(1, num_domains + 1):
        ax_amp.axvline(x=i * Lc, color="gray", linestyle=":", alpha=0.5)
        if i == 1:
            ax_amp.text(i * Lc, max(amp_mm) * 1.1, " $L_c$", color="black")

    ax_amp.set_title("Amplitude Growth vs Distance", fontsize=14)
    ax_amp.set_xlabel("Distance z", fontsize=12)
    ax_amp.set_ylabel("Electric Field Amplitude |E(z)|", fontsize=12)
    ax_amp.legend(fontsize=10)
    ax_amp.grid(True, linestyle=":", alpha=0.6)

    # --- REMOVE NUMBERS FROM AXES ---
    ax_amp.set_xticks([])
    ax_amp.set_yticks([])

    plt.tight_layout()
    output_path = "shg_phasor_diagram_no_numbers.png"
    plt.savefig(output_path, dpi=300)
    return output_path


plot_visualizations()
