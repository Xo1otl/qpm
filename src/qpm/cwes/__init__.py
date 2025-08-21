import jax.numpy as jnp
from jax import jit, lax


@jit
def get_L(delta_k1: jnp.ndarray, delta_k2: jnp.ndarray) -> jnp.ndarray:
    """線形演算子Lを生成"""
    return jnp.array([0.0, delta_k1 * 1j, (delta_k1 + delta_k2) * 1j], dtype=jnp.complex64)


@jit
def phi(omega: jnp.ndarray, h: float) -> jnp.ndarray:
    """
    IPM1予測子で使用される積分関数 Φ(Ω, h) = (e^(Ωh) - 1) / Ω を計算する。
    JAXのjitに対応するため、jnp.whereで条件分岐を処理する。
    """
    is_small = jnp.abs(omega) < 1e-9
    val_small = h + h**2 * omega / 2.0
    val_large = (jnp.exp(omega * h) - 1.0) / omega
    return jnp.where(is_small, val_small, val_large)


@jit
def predictor_ipm1(B_in: jnp.ndarray, h: float, kappa_val: float, L: jnp.ndarray) -> jnp.ndarray:
    """
    IPM1 (Interaction Picture Method 1st order) スキームの1ステップを計算する。
    """
    B1n, B2n, B3n = B_in
    L1, L2, L3 = L

    # exp(L*h) の計算を一度だけ行い、結果を再利用する
    exp_Lh = jnp.exp(L * h)
    exp_L1h, exp_L2h, exp_L3h = exp_Lh

    omega_a = L2 - 2 * L1
    omega_b = L3 - L2 - L1

    delta_B_NL1 = 1j * kappa_val * exp_L1h * (
        jnp.conj(B1n) * B2n * phi(omega_a, h) +
        jnp.conj(B2n) * B3n * phi(omega_b, h)
    )
    delta_B_NL2 = 1j * kappa_val * exp_L2h * (
        B1n**2 * phi(-omega_a, h) +
        2 * jnp.conj(B1n) * B3n * phi(omega_b, h)
    )
    delta_B_NL3 = 1j * 3 * kappa_val * exp_L3h * (
        B1n * B2n * phi(-omega_b, h)
    )
    delta_B_NL = jnp.array([delta_B_NL1, delta_B_NL2, delta_B_NL3])

    B_pred = exp_Lh * B_in + delta_B_NL
    return B_pred


def simulate_twm(superlattice: jnp.ndarray, delta_k1: jnp.ndarray, delta_k2: jnp.ndarray, B_initial: jnp.ndarray) -> jnp.ndarray:
    """指定された単一の波長、温度、格子で伝播を計算する。"""
    if superlattice.ndim != 2 or superlattice.shape[1] != 2:
        raise ValueError(
            "スーパーラティス構造を定義する配列の形状は (N, 2) で、Nがドメインの数、各行が [h, kappa] でなければなりません。")
    if B_initial.shape != (3,) or B_initial.dtype != jnp.complex64:
        raise ValueError(
            "B_initial must be a 1D array of shape (3,) and dtype jnp.complex64")

    L = get_L(delta_k1, delta_k2)

    def ipm1_scan_step(B_carry, domain):
        h, kappa_val = domain
        B_next = predictor_ipm1(B_carry, h, kappa_val, L)
        return B_next, None

    B_final, _ = lax.scan(ipm1_scan_step, B_initial, superlattice)

    return B_final
