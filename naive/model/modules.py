import jax
import jax.lax as lax
import jax.numpy as jnp

from typing import Literal


class ReLU:

    def __call__(self, x):
        return jnp.where(x > 0, x, 0.0)


class Embeddings:

    def __init__(
        self,
        random_key,
        num_embeds: int,
        embed_dim: int,
    ):
        self.weights = jax.random.normal(random_key, (num_embeds, embed_dim))

    def __call__(self, i):
        return self.weights[i]


class Linear:

    def __init__(
        self,
        random_key,
        in_channels: int,
        out_channels: int,
    ):
        self.weights = jax.random.normal(random_key, (in_channels, out_channels))
        self.bias = jnp.zeros((out_channels,))
    
    def __call__(self, x):
        x = jnp.dot(x, self.weights)
        x = x + self.bias
        return x


class Conv2D:

    def __init__(
        self,
        random_key,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        padding: Literal["SAME", "VALID"],
    ):
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        self.weights = jax.random.normal(
            random_key,
            (out_channels, in_channels, *self.kernel_size),
            dtype=jnp.float32,
        )
        self.bias = jnp.zeros((out_channels,))

    def __call__(self, x):
        x = lax.conv_general_dilated(
            x,
            self.weights,
            window_strides=(self.stride, self.stride),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        x = x + self.bias
        return x
