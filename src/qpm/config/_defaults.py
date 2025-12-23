from qpm import ape, cwes, mgoslt, wgmode


def new_kappa_config() -> cwes.KappaConfig:
    """
    Creates a Kappa configuration with the grid defaults from calculate_kappa.py.
    """
    return cwes.KappaConfig(
        fund_wavelength=1.031,
        shg_wavelength=0.5155,
        x_min=-50.0,
        x_max=50.0,
        y_min=-50.0,
        y_max=50.0,
        nx=500,
        ny=500,
        d33_val=1.38e-5,
    )


def new_process_params() -> ape.ProcessParams:
    """Creates default process parameters with buried structure enabled."""
    return ape.ProcessParams(
        temp_c=70.0,
        d_pe_coeff=0.045,
        t_pe_hours=8.0,
        mask_width_um=50.0,
        t_anneal_hours=100.0,
        d_x_coeff=1.3,
        d_y_coeff=1.3 / 1.5,
        is_buried=True,
    )


def new_simulation_config(wavelength_um: float, process_params: ape.ProcessParams | None = None) -> wgmode.SimulationConfig:
    """
    Creates a simulation configuration with the geometry defaults from calculate_kappa.py.
    """
    process_params = process_params if process_params is not None else new_process_params()
    return wgmode.SimulationConfig(
        wavelength_um=wavelength_um,
        width_min=-50.0,
        width_max=50.0,
        depth_min=-50.0,
        depth_max=50.0,
        core_resolution=0.5,
        cladding_resolution=1.0,
        core_width_half=10.0,
        core_depth_max=15.0,
        core_distance=2.0,
        num_modes=2,
        plot_modes=False,
        n_guess_offset=5e-3,
        process_params=process_params,
        upper_cladding_n=1.0,
        apply_upper_cladding=None,
        n_sub=mgoslt.sellmeier_n_eff(wavelength_um, process_params.temp_c),
        delta_n0=ape.get_delta_n0(wavelength_um),
    )
