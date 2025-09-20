import jax.numpy as jnp
from typing import Literal

from .modules import ReLU, Conv2D


class ConvBlock:
    """Aggregated conv2d blocks"""

    def __init__(
        self,
        random_key,
        kernel_size: int,
        channels: list[int],
        stride: int,
        padding: Literal["SAME", "VALID"],
        last_channel: int = None,
        last_stride: int = None,
        last_padding: Literal["SAME", "VALID"] = None,
    ):
        last_channel = channels[-1] if last_channel is None else last_channel
        last_stride = stride if last_stride is None else last_stride
        last_padding = padding if last_padding is None else last_padding

        self.convs = [
            Conv2D(
                random_key,
                kernel_size,
                in_channels=channels[i],
                out_channels=channels[i + 1],
                stride=stride,
                padding=padding,
            )
            for i in range(len(channels) - 1)
        ]
        self.last_conv = Conv2D(
            random_key,
            kernel_size,
            in_channels=channels[-1],
            out_channels=last_channel,
            stride=last_stride,
            padding=last_padding,
        )

    def __call__(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.last_conv(x)
        return x


class UNetBlock:

    def __init__(
        self,
        random_key,
        num_blocks: int,
        kernel_sizes: list[int],
        channels: list[list[int]],
        strides: list[int],
        paddings: Literal["SAME", "VALID"],
        return_features: bool = False,
        last_channels: list[int] = None,
        last_strides: list[int] = None,
        last_paddings: list[Literal["SAME", "VALID"]] = None,
    ):
        self.num_blocks = num_blocks
        self.return_features = return_features

        self.blocks = [
            ConvBlock(
                random_key,
                kernel_sizes[b],
                channels=channels[b],
                stride=strides[b],
                padding=paddings[b],
                last_channel=last_channels[b] if last_channels is not None else None,
                last_stride=last_strides[b] if last_strides is not None else None,
                last_padding=last_paddings[b] if last_paddings is not None else None,
            )
            for b in range(num_blocks)
        ]

    def __call__(self, x):
        cached_x = []
        for block in self.blocks:
            x = block(x)
            if self.return_features:
                cached_x.append(x)
        if self.return_features:
            return x, cached_x
        return x


class UNet:

    def __init__(self, random_key, down_args, middle_args, up_args, conv_args):
        self.down_block = UNetBlock(random_key=random_key, return_features=True, **down_args)
        self.middle_block = UNetBlock(random_key=random_key, **middle_args)
        self.up_block = UNetBlock(random_key=random_key, **up_args)
        self.last_convs = ConvBlock(random_key=random_key, **conv_args)

    def __call__(self, x):
        x, cached_x = self.down_block(x)
        x = self.middle_block(x)

        up_block_inputs = jnp.concatenate([x, cached_x], axis=-1)
        x = self.up_block(up_block_inputs)

        x = self.last_convs(x)

        return x
