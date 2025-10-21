from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

# --- Type Definitions ---
DomainIndexes = jax.Array  # shape: (num_domains,) -> [0, 1, ..., N-1]
WidthFn = Callable[[DomainIndexes], jax.Array]
KappaFn = Callable[[DomainIndexes], jax.Array]


class Grating(NamedTuple):
    """
    A data structure containing the physical properties of a grating.
    This is the primary output of the build process.

    Attributes:
        widths: 1D array of domain widths.
        kappas: 1D array of corresponding kappa values.
    """

    widths: jax.Array
    kappas: jax.Array


# --- Profile Definition (Declarative Blueprint) ---
@dataclass(frozen=True)
class Profile:
    """A declarative mathematical blueprint for a single grating segment."""

    num_domains: int
    width_fn: WidthFn
    kappa_fn: KappaFn


# --- Profile Factory Functions ---
def uniform_profile(num_domains: int, period: float, kappa_mag: float, start_sign: float = 1.0) -> Profile:
    """
    Creates a Profile for a grating with a uniform period.
    Provides the most common alternating-sign behavior by default.

    Args:
        num_domains: The number of domains in the segment.
        period: The grating period (width of two adjacent domains).
        kappa_mag: The magnitude of the coupling coefficient.
        start_sign: The sign of kappa for the first domain (+1.0 or -1.0).
    """
    domain_width = period / 2.0

    def kappa_fn(i: DomainIndexes) -> jax.Array:
        # Alternating signs for kappa
        signs = jnp.power(-1.0, i)
        return jnp.sign(start_sign) * jnp.full_like(i, kappa_mag, dtype=jnp.float32) * signs

    return Profile(
        num_domains=num_domains,
        width_fn=lambda i: jnp.full_like(i, domain_width, dtype=jnp.float32),
        kappa_fn=kappa_fn,
    )


def tapered_profile(num_domains: int, start_width: float, chirp_rate: float, kappa_mag: float, start_sign: float = 1.0) -> Profile:
    """
    Creates a Profile for a tapered (chirped) grating.
    Provides the most common alternating-sign behavior by default.

    Args:
        num_domains: The number of domains in the segment.
        start_width: The width of the initial domain.
        chirp_rate: The rate at which the domain width changes.
        kappa_mag: The magnitude of the coupling coefficient.
        start_sign: The sign of kappa for the first domain (+1.0 or -1.0).
    """

    def width_fn(i: DomainIndexes) -> jax.Array:
        return start_width / jnp.sqrt(1 + 2 * chirp_rate * start_width * i)

    def kappa_fn(i: DomainIndexes) -> jax.Array:
        # Alternating signs for kappa
        signs = jnp.power(-1.0, i)
        return jnp.sign(start_sign) * jnp.full_like(i, kappa_mag, dtype=jnp.float32) * signs

    return Profile(num_domains=num_domains, width_fn=width_fn, kappa_fn=kappa_fn)


# --- Grating Construction ---
def _build_from_profile(profile: Profile) -> Grating:
    """[Internal] Realizes a single Profile into a concrete Grating structure."""
    if profile.num_domains == 0:
        empty_arr = jnp.array([], dtype=jnp.float32)
        return Grating(widths=empty_arr, kappas=empty_arr)

    indices = jnp.arange(profile.num_domains, dtype=jnp.float32)
    widths = profile.width_fn(indices)
    kappas = profile.kappa_fn(indices)
    return Grating(widths=widths, kappas=kappas)


def build(profiles: Profile | list[Profile]) -> Grating:
    """
    Builds a complete Grating from one or more Profile blueprints.
    If multiple profiles are provided, they are concatenated in order.
    """
    if not isinstance(profiles, list):
        profiles = [profiles]

    if not profiles:
        empty_arr = jnp.array([], dtype=jnp.float32)
        return Grating(widths=empty_arr, kappas=empty_arr)

    # Realize each profile into a concrete Grating segment
    segments = [_build_from_profile(p) for p in profiles if p.num_domains > 0]

    if not segments:
        empty_arr = jnp.array([], dtype=jnp.float32)
        return Grating(widths=empty_arr, kappas=empty_arr)

    # Concatenate the widths and kappas from all segments
    all_widths = jnp.concatenate([s.widths for s in segments])
    all_kappas = jnp.concatenate([s.kappas for s in segments])

    return Grating(widths=all_widths, kappas=all_kappas)
