from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

# --- 型定義 ---
Section = jnp.ndarray  # shape: (num_domains, 2) -> [width, kappa]
DomainIndexes = jnp.ndarray  # shape: (num_domains,) -> [0, 1, ..., N-1]
Batch = jnp.ndarray  # shape: (batch_size, num_domains, 2)
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
def _realize(profile: Profile) -> Section:
    """【内部関数】単一のProfileを具体的なJAX配列に変換する"""
    if profile.num_domains == 0:
        return jnp.zeros((0, 2), dtype=jnp.float32)

    indices = jnp.arange(profile.num_domains, dtype=jnp.float32)
    widths = profile.width_fn(indices)
    kappas = profile.kappa_fn(indices)
    return jnp.stack([widths, kappas], axis=1)


def build(profiles: Profile | list[Profile]) -> Section:
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


# --- バッチ生成関数 ---
def generate_tapered_batch(num_domains: int, start_widths: list[float], chirp_rates: list[float], kappa_mag: float) -> Batch:
    """vmapを使ってテーパー状グレーティングのバッチを生成する"""

    @jax.jit
    def make_one_grating(params: dict[str, float]) -> Section:
        # kappa_magをkappaに変更
        profile = tapered_profile(num_domains=num_domains, start_width=params["start_width"], chirp_rate=params["chirp_rate"], kappa_mag=kappa_mag)
        return build(profile)

    sw_grid, cr_grid = jnp.meshgrid(jnp.array(start_widths), jnp.array(chirp_rates), indexing="ij")
    params_batch = {"start_width": sw_grid.ravel(), "chirp_rate": cr_grid.ravel()}
    return jax.vmap(make_one_grating)(params_batch)


if __name__ == "__main__":
    # --- BENCHMARK 1: Uniform Grating ---
    print("--- Benchmark 1: Uniform Grating ---")
    # kappa_magをkappaに変更
    profile1 = uniform_profile(num_domains=10, period=3.0, kappa_mag=1.5)
    uniform_grating = build(profile1)
    print("Uniform grating segment (first 5 domains):\n", uniform_grating[:5])
    print("-" * 40)

    # --- BENCHMARK 2: Tapered Grating ---
    print("--- Benchmark 2: Tapered Grating ---")
    # kappa_magをkappaに変更
    profile2 = tapered_profile(num_domains=10, start_width=3.6, chirp_rate=0.01, kappa_mag=1.0)
    tapered_grating = build(profile2)
    print("Tapered grating segment (first 5 domains):\n", tapered_grating[:5])
    print("-" * 40)

    # --- ARCHITECTURE DEMO 1: Composite Grating ---
    print("--- Architecture Demo: Composite Grating ---")
    # ファクトリ関数に start_sign を追加したことで、
    # 連結時の符号制御がより直感的になった。
    profile_a = uniform_profile(num_domains=4, period=4.0, kappa_mag=1.0)

    # 前のセグメントのドメイン数が偶数なら次は+1、奇数なら-1から始める
    next_start_sign = jnp.power(-1.0, profile_a.num_domains)

    profile_b = tapered_profile(
        num_domains=5,
        start_width=3.6,
        chirp_rate=0.01,
        kappa_mag=1.2,
        start_sign=float(next_start_sign),  # 計算した開始符号を渡す
    )

    composite_grating = build([profile_a, profile_b])
    print("Composite grating shape:", composite_grating.shape)
    print("Note the sign flip at the boundary between segments (domain 4 to 5):")
    # 4番目のドメイン (i=3) は (-1)^3 = 負
    # 5番目のドメイン (i=0 in profile_B) は start_sign=+1 なので 正
    print(composite_grating[3:6])
    print("-" * 40)

    # --- ARCHITECTURE DEMO 2: Batched Generation ---
    print("--- Architecture Demo: Batched Generation ---")
    # kappa_magをkappaに変更
    batched_gratings = generate_tapered_batch(num_domains=100, start_widths=[3.2, 3.6], chirp_rates=[0.01, 0.02], kappa_mag=1.0)
    print(f"Generated a batch of gratings with shape: {batched_gratings.shape}")
    print("(Batch size of 4 = 2 start_widths * 2 chirp_rates)")
