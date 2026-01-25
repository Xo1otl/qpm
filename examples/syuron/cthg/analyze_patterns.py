import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from analyze_checkpoint import SimulationConfig


def main():
    pkl_path = "optimized_0.5_final.pkl"
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    widths = np.abs(np.array(data["params"]))
    n_domains = len(widths)

    # 1. Envelope Analysis
    # Uppper Envelope: find_peaks
    peaks, _ = scipy.signal.find_peaks(widths, distance=10)
    # Lower Envelope: find_peaks on inverted signal
    valleys, _ = scipy.signal.find_peaks(-widths, distance=10)

    # Interpolate envelopes
    upper_env = np.interp(np.arange(n_domains), peaks, widths[peaks])
    lower_env = np.interp(np.arange(n_domains), valleys, widths[valleys])

    # Moving Average (Trend)
    window = 100
    moving_avg = np.convolve(widths, np.ones(window) / window, mode="same")

    # 2. Spectrogram (STFT)
    # "Signal" is width sequence w[n]
    f, t, Zxx = scipy.signal.stft(widths, fs=1.0, nperseg=256, noverlap=200)

    # 3. FFT
    fft_vals = np.fft.rfft(widths - np.mean(widths))
    fft_freq = np.fft.rfftfreq(n_domains, d=1.0)

    # Plotting
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)

    # A. Widths with Envelope
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(widths, color="gray", alpha=0.5, label="Raw Widths", linewidth=0.5)
    ax1.plot(upper_env, color="red", linestyle="--", label="Upper Envelope")
    ax1.plot(lower_env, color="blue", linestyle="--", label="Lower Envelope")
    ax1.plot(moving_avg, color="green", linewidth=2, label="Moving Average (100)")
    ax1.set_title("Domain Widths with Envelopes")
    ax1.set_xlabel("Domain Index")
    ax1.set_ylabel("Width (um)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # B. Spectrogram
    ax2 = fig.add_subplot(gs[1, :])
    # Zxx is complex, take magnitude
    # Pcolormesh need t and f.
    # t is segment center index. f is frequency (cycles/sample)
    # 0.5 cycles/sample = period 2 (Nyquist).
    pcm = ax2.pcolormesh(t, f, np.abs(Zxx), shading="gouraud", cmap="inferno")
    ax2.set_title("Spectrogram (Evolution of Spatial Frequencies)")
    ax2.set_ylabel("Frequency (1/Index)")
    ax2.set_xlabel("Domain Index")
    fig.colorbar(pcm, ax=ax2, label="Magnitude")

    # C. FFT
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(fft_freq, np.abs(fft_vals))
    ax3.set_title("Global FFT Spectrum")
    ax3.set_xlabel("Frequency (1/Index)")
    ax3.set_ylabel("Magnitude")
    ax3.set_xlim(0, 0.5)
    ax3.grid(True, alpha=0.3)

    # Annotate potential Lc freqs
    # Lc_SHG ~ 3.96. If strictly alternating +/- Lc... Width = 3.96. Period = 2 samples? No.
    # Period in terms of INDEX?
    # If the pattern is w1, w2, w1, w2... period is 2. Freq = 0.5.
    # If width is constant, signal is DC (+ noise).
    # But width itself VARIES.
    # For SHG: Widths ~ 3.96. If constant, it's DC line at 3.96.
    # For SFG: Widths ~ 1.10. It's DC line at 1.10.
    # The VARIATION is what we see in FFT.
    # The spectrogram basically shows the transition from DC-like 3.96 to DC-like 1.10?
    # No, because "width vs index" handles the DC offset.
    # The "Fanning" implies modulation: w[n] = Base + A*sin(kn).

    # D. Envelope Width (Upper - Lower)
    ax4 = fig.add_subplot(gs[2, 1])
    envelope_width = upper_env - lower_env
    ax4.plot(envelope_width, color="purple")
    ax4.set_title("Modulation Amplitude (Envelope Width)")
    ax4.set_xlabel("Domain Index")
    ax4.set_ylabel("Delta Width (um)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pattern_analysis.png")
    print("Saved pattern_analysis.png")


if __name__ == "__main__":
    main()
