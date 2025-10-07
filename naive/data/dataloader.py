import math
import jax.numpy as jnp

from jax import Array
from typing import Iterable


class DataLoader:

    def __init__(self, dataset, batch_size: int):

        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    
    def __iter__(self) -> Iterable[tuple[Array, Array]]:
        for i in range(0, len(self.dataset), self.batch_size):
            samples = self.dataset[i : i + self.batch_size]
            images, labels = list(zip(samples))
            images = jnp.array(images, dtype=jnp.float32)
            labels = jnp.array(labels, dtype=jnp.int32)
            yield images, labels
