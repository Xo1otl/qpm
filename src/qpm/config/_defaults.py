from qpm import ape


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
