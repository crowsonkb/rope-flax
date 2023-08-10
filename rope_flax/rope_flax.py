"""Rotary Position Embedding for Flax."""

from functools import wraps
from typing import Tuple

from einshape import jax_einshape as einshape
import flax.linen as nn
import jax
import jax.numpy as jnp


def rotate_half(x: jax.Array) -> jax.Array:
    x = einshape("...(dr)->...dr", x, r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack((-x2, x1), axis=-1)
    return einshape("...dr->...(dr)", x)


def apply_rotary_emb(
    freqs: jax.Array, t: jax.Array, start_index: int = 0, scale: float = 1.0
) -> jax.Array:
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    if end_index > t.shape[-1]:
        raise ValueError(
            f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        )
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * jnp.cos(freqs) * scale) + (rotate_half(t) * jnp.sin(freqs) * scale)
    return jnp.concatenate((t_left, t, t_right), axis=-1)


def freqs_lang(theta: float = 10000.0) -> callable:
    @wraps(freqs_lang)
    def init(key, shape, dtype=jnp.float32):
        dim = shape[-1] * 2
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=dtype)[: (dim // 2)] / dim))
        return jnp.broadcast_to(freqs, shape)

    return init


def freqs_pixel(max_freq: float = 256.0) -> callable:
    @wraps(freqs_pixel)
    def init(key, shape, dtype=jnp.float32):
        freqs = jnp.linspace(1.0, max_freq / 2, shape[-1], dtype=dtype) * jnp.pi
        return jnp.broadcast_to(freqs, shape)

    return init


class RoPE(nn.Module):
    dim: int
    num_heads: int = 1
    start_index: int = 0
    dtype: jnp.dtype = jnp.float32
    freqs_init: callable = freqs_lang()

    def setup(self):
        shape = self.num_heads, self.dim // 2
        self.freqs = self.param("freqs", self.freqs_init, shape)

    def get_freqs(self, pos: jax.Array) -> jax.Array:
        freqs = jnp.repeat(self.freqs, 2, axis=-1)
        return pos[..., None, None] * freqs.astype(self.dtype)

    def __call__(self, x: jax.Array, pos: jax.Array) -> jax.Array:
        freqs = self.get_freqs(pos)
        return apply_rotary_emb(freqs, x, start_index=self.start_index)


def centers(start: float, stop: float, num: int, dtype: jnp.dtype = None) -> jax.Array:
    edges = jnp.linspace(start, stop, num + 1, dtype=dtype)
    return (edges[:-1] + edges[1:]) / 2


def make_grid(h_pos: jax.Array, w_pos: jax.Array) -> jax.Array:
    grid = jnp.stack(jnp.meshgrid(h_pos, w_pos, indexing="ij"), axis=-1)
    return einshape("hwd->(hw)d", grid)


def bounding_box(h: int, w: int, pixel_aspect_ratio: float = 1.0) -> Tuple[int, int, int, int]:
    # Adjusted dimensions
    w_adj = w * pixel_aspect_ratio
    h_adj = h

    # Adjusted aspect ratio
    ar_adj = w_adj / h_adj

    # Determine bounding box based on the adjusted aspect ratio
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj

    return y_min, y_max, x_min, x_max


def make_axial_pos(
    h: int, w: int, pixel_aspect_ratio: float = 1.0, align_corners: bool = False
) -> jax.Array:
    y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    if align_corners:
        return make_grid(jnp.linspace(y_min, y_max, h), jnp.linspace(x_min, x_max, w))
    else:
        return make_grid(centers(y_min, y_max, h), centers(x_min, x_max, w))


class AxialRoPE(nn.Module):
    dim: int
    num_heads: int = 1
    start_index: int = 0
    dtype: jnp.dtype = jnp.float32
    freqs_h_init: callable = freqs_pixel()
    freqs_w_init: callable = freqs_pixel()

    def setup(self):
        shape = self.num_heads, self.dim // 4
        self.freqs_h = self.param("freqs_h", self.freqs_h_init, shape)
        self.freqs_w = self.param("freqs_w", self.freqs_w_init, shape)

    def get_freqs(self, pos: jax.Array) -> jax.Array:
        if pos.shape[-1] != 2:
            raise ValueError("input shape must be (..., 2)")
        freqs_h = pos[..., None, None, 0] * self.freqs_h.astype(self.dtype)
        freqs_w = pos[..., None, None, 1] * self.freqs_w.astype(self.dtype)
        freqs = jnp.concatenate((freqs_h, freqs_w), axis=-1)
        freqs = jnp.repeat(freqs, 2, axis=-1)
        return freqs

    def __call__(self, x: jax.Array, pos: jax.Array) -> jax.Array:
        freqs = self.get_freqs(pos)
        return apply_rotary_emb(freqs, x, start_index=self.start_index)
