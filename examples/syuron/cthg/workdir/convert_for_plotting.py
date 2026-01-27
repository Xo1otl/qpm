import pickle
import numpy as np


# Dummy class to handle unpickling
class SimulationConfig:
    pass


def main():
    input_file = "optimized_best2.pkl"
    output_file = "optimized_best2_plot.pkl"

    print(f"Loading {input_file}...")
    try:
        with open(input_file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

    # Extract relevant data
    # The target script expects 'widths' and 'amp'
    output_data = {}

    if isinstance(data, dict):
        if "params" in data:
            output_data["widths"] = np.array(data["params"])
        else:
            print("Error: 'params' key not found in data.")
            return

        if "thg_amp" in data:
            output_data["amp"] = data["thg_amp"]
        else:
            print("Warning: 'thg_amp' key not found, setting to 0.0")
            output_data["amp"] = 0.0
    else:
        print("Error: loaded data is not a dictionary.")
        return

    print(f"Extracted widths shape: {output_data['widths'].shape}")
    print(f"Extracted amp: {output_data['amp']}")

    print(f"Saving to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(output_data, f)

    print("Conversion complete.")


if __name__ == "__main__":
    main()
