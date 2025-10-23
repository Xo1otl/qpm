import jax
import jax.numpy as jnp
from jax import jit, lax

OMEGA_SMALL_EPS: float = 1e-9
SUPERLATTICE_SPEC: tuple[str, tuple[str, str]] = ("domain", ("h", "kappa"))


@jit
def get_lin(delta_k1: jax.Array, delta_k2: jax.Array) -> jax.Array:
    """線形演算子Lを生成"""
    return jnp.array([0.0, delta_k1 * 1j, (delta_k1 + delta_k2) * 1j], dtype=jnp.complex64)


@jit
def phi(omega: jax.Array, h: float) -> jax.Array:
    """
    ETD予測子で使用される積分関数 Φ(Ω, h) = (e^(Ωh) - 1) / Ω を計算する。
    JAXのjitに対応するため、jnp.whereで条件分岐を処理する。
    """
    is_small = jnp.abs(omega) < OMEGA_SMALL_EPS
    val_small = h + h**2 * omega / 2.0
    val_large = (jnp.exp(omega * h) - 1.0) / omega
    return jnp.where(is_small, val_small, val_large)


@jit
def propagate_domain(b_in: jax.Array, h: float, kappa_val: float, lin: jax.Array) -> jax.Array:
    """
    ETD スキームの1ステップを計算する。
    """
    b1n, b2n, b3n = b_in
    lin1, lin2, lin3 = lin

    # exp(L*h) の計算を一度だけ行い、結果を再利用する
    exp_linh = jnp.exp(lin * h)
    exp_lin1h, exp_lin2h, exp_lin3h = exp_linh

    omega_a = lin2 - 2 * lin1
    omega_b = lin3 - lin2 - lin1

    delta_b_nlin1 = 1j * kappa_val * exp_lin1h * (jnp.conj(b1n) * b2n * phi(omega_a, h) + jnp.conj(b2n) * b3n * phi(omega_b, h))
    delta_b_nlin2 = 1j * kappa_val * exp_lin2h * (b1n**2 * phi(-omega_a, h) + 2 * jnp.conj(b1n) * b3n * phi(omega_b, h))
    delta_b_nlin3 = 1j * 3 * kappa_val * exp_lin3h * (b1n * b2n * phi(-omega_b, h))
    delta_b_nlin = jnp.array([delta_b_nlin1, delta_b_nlin2, delta_b_nlin3])

    return exp_linh * b_in + delta_b_nlin


def simulate_twm(
    domain_widths: jax.Array,
    kappa_vals: jax.Array,
    delta_k1: jax.Array,
    delta_k2: jax.Array,
    b_initial: jax.Array,
) -> jax.Array:
    """指定された単一の波長、温度、格子で伝播を計算する。"""
    if domain_widths.ndim != 1 or kappa_vals.ndim != 1 or domain_widths.shape != kappa_vals.shape:
        msg = "domain_widths and kappa_vals must be 1D arrays of the same shape."
        raise ValueError(msg)
    if b_initial.shape != (3,) or b_initial.dtype != jnp.complex64:
        msg = "b_initial must be a 1D array of shape (3,) and dtype jnp.complex64"
        raise ValueError(msg)

    lin = get_lin(delta_k1, delta_k2)

    def propagator_step(b_carry: jax.Array, domain_params: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, None]:
        h, kappa_val = domain_params
        b_next = propagate_domain(b_carry, h, kappa_val, lin)
        return b_next, None

    b_final, _ = lax.scan(propagator_step, b_initial, (domain_widths, kappa_vals))

    return b_final
