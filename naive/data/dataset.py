import numpy as np

from typing import Literal

from .utils import load_cifar100, load_meta


class CIFAR100Dataset:

    def __init__(
        self,
        cifa100_dir: str,
        split: Literal["train", "test"],
        random_key,
    ):

        self.images, _, self.coarse_labels = load_cifar100(cifa100_dir, kind=split)
        self.num_samples = self.images.shape[0]
        shuffled_ids = np.array(list(range(self.num_samples)))
        np.random.shuffle(shuffled_ids)

        self.images = self.images[shuffled_ids]
        self.coarse_labels = self.coarse_labels[shuffled_ids]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index):
        return self.images[index], self.coarse_labels[index]
