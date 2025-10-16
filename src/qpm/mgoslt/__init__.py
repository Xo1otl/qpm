import jax.numpy as jnp
from jax import jit

SELLMEIER_PARAMS = {
    "a": jnp.array([4.5615, 0.08488, 0.1927, 5.5832, 8.3067, 0.021696]),
    "b": jnp.array([4.782e-07, 3.0913e-08, 2.7326e-08, 1.4837e-05, 1.3647e-07]),
}


@jit
def sellmeier_n_eff(wl: jnp.ndarray, temp: jnp.ndarray) -> jnp.ndarray:
    """Sellmeier方程式を用いて実効屈折率を計算"""
    f = (temp - 24.5) * (temp + 24.5 + 2.0 * 273.16)
    lambda_sq = wl**2
    a, b = SELLMEIER_PARAMS["a"], SELLMEIER_PARAMS["b"]
    n_sq = (
        a[0]
        + b[0] * f
        + (a[1] + b[1] * f) / (lambda_sq - (a[2] + b[2] * f) ** 2)
        + (a[3] + b[3] * f) / (lambda_sq - (a[4] + b[4] * f) ** 2)
        - a[5] * lambda_sq
    )
    return jnp.sqrt(n_sq)


@jit
def calc_twm_delta_k(wl1: jnp.ndarray, wl2: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Three-Wave Mixingの位相不整合量を計算"""
    wl_sum = (wl1 * wl2) / (wl1 + wl2)
    n1, n2, n_sum = (
        sellmeier_n_eff(wl1, t),
        sellmeier_n_eff(wl2, t),
        sellmeier_n_eff(wl_sum, t),
    )
    return 2.0 * jnp.pi * (n_sum / wl_sum - n1 / wl1 - n2 / wl2)
