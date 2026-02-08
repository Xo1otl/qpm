import numpy as np


def trace_error_diffusion():
    period = 8
    duties = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    error = 0.0

    print(f"{'Duty':<6} {'Ideal':<6} {'PrevErr':<8} {'Target':<6} {'Rounded':<8} {'NewErr':<6}")
    print("-" * 50)

    for d in duties:
        ideal_width = d * period
        target_val = ideal_width + error
        # Round to nearest even number (scale of 2)
        w_after = int(np.round(target_val / 2) * 2)
        w_after = max(0, min(period, w_after))

        new_error = target_val - w_after

        print(f"{d:<6.1f} {ideal_width:<6.1f} {error:<8.2f} {target_val:<6.2f} {w_after:<8d} {new_error:<6.2f}")

        error = new_error


if __name__ == "__main__":
    trace_error_diffusion()
