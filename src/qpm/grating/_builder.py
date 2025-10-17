from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

# --- 型定義 ---
Grating = jnp.ndarray  # shape: (num_domains, 2) -> [width, kappa]
DomainIndexes = jnp.ndarray  # shape: (num_domains,) -> [0, 1, ..., N-1]
WidthFn = Callable[[DomainIndexes], jnp.ndarray]
KappaFn = Callable[[DomainIndexes], jnp.ndarray]


# --- データ構造とファクトリ関数 ---
@dataclass(frozen=True)
class Profile:
    """単一グレーティングセグメントの宣言的な数学的設計図 (データコンテナ)"""

    num_domains: int
    width_fn: WidthFn
    kappa_fn: KappaFn


def uniform_profile(num_domains: int, period: float, kappa_mag: float, start_sign: float = 1.0) -> Profile:
    """
    一様な周期を持つグレーティングのProfileを生成する。
    最も一般的な交互反転の振る舞いをデフォルトで提供する。

    Args:
        num_domains: ドメイン数。
        period: グレーティング周期。
        kappa: 結合係数の大きさ。
        start_sign: 最初のドメインのkappaの符号 (+1.0 または -1.0)。
    """
    domain_width = period / 2.0

    def kappa_fn(i: DomainIndexes) -> jnp.ndarray:
        signs = jnp.power(-1.0, i)
        return jnp.sign(start_sign) * jnp.full_like(i, kappa_mag, dtype=jnp.float32) * signs

    return Profile(
        num_domains,
        lambda i: jnp.full_like(i, domain_width, dtype=jnp.float32),
        kappa_fn,
    )


def tapered_profile(num_domains: int, start_width: float, chirp_rate: float, kappa_mag: float, start_sign: float = 1.0) -> Profile:
    """
    テーパー状グレーティングのProfileを生成する。
    最も一般的な交互反転の振る舞いをデフォルトで提供する。

    Args:
        num_domains: ドメイン数。
        start_width: 開始ドメイン幅。
        chirp_rate: チャープ率。
        kappa: 結合係数の大きさ。
        start_sign: 最初のドメインのkappaの符号 (+1.0 または -1.0)。
    """

    def width_fn(i: DomainIndexes) -> jnp.ndarray:
        return start_width / jnp.sqrt(1 + 2 * chirp_rate * start_width * i)

    def kappa_fn(i: DomainIndexes) -> jnp.ndarray:
        signs = jnp.power(-1.0, i)
        return jnp.sign(start_sign) * jnp.full_like(i, kappa_mag, dtype=jnp.float32) * signs

    return Profile(num_domains, width_fn, kappa_fn)


# --- グレーティングを構築する関数 ---
def _realize(profile: Profile) -> Grating:
    """【内部関数】単一のProfileを具体的なJAX配列に変換する"""
    if profile.num_domains == 0:
        return jnp.zeros((0, 2), dtype=jnp.float32)

    indices = jnp.arange(profile.num_domains, dtype=jnp.float32)
    widths = profile.width_fn(indices)
    kappas = profile.kappa_fn(indices)
    return jnp.stack([widths, kappas], axis=1)


def build(profiles: Profile | list[Profile]) -> Grating:
    """
    単一または複数のProfileからグレーティングを構築する。
    連結時の符号の連続性は、ユーザーが kappa_fn を定義することで保証する必要がある。
    """
    if not isinstance(profiles, list):
        profiles = [profiles]

    if not profiles:
        return jnp.zeros((0, 2), dtype=jnp.float32)

    # 符号管理ロジックが不要になり、単純な連結処理になる
    segments = [_realize(p) for p in profiles if p.num_domains > 0]

    if not segments:
        return jnp.zeros((0, 2), dtype=jnp.float32)

    return jnp.concatenate(segments, axis=0)
