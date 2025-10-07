import os
import pickle
import numpy as np
from typing import Tuple, List, Dict


def load_cifar100(data_dir: str, kind: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIFAR-100 dataset.

    Args:
        data_dir: Path to directory containing the CIFAR-100 python version files.
            e.g. the file 'train' and 'test'.
        kind: 'train' or 'test'.

    Returns:
        images: numpy array of shape (N, 32, 32, 3), dtype uint8
        fine_labels: numpy array of shape (N,), dtype int
        coarse_labels: numpy array of shape (N,), dtype int
    """
    assert kind in ["train", "test"], "kind must be 'train' or 'test'"
    file_path = os.path.join(data_dir, kind)
    with open(file_path, "rb") as f:
        # The CIFAR-100 python version uses pickle; in Python 3 need encoding='bytes'
        data_dict = pickle.load(f, encoding="bytes")

    # Keys are b'data', b'fine_labels', b'coarse_labels'
    # Also b'filenames', etc.
    data = data_dict[b"data"]  # shape (N, 3072), uint8
    fine = np.array(data_dict[b"fine_labels"], dtype=np.int64)  # shape (N,)
    coarse = np.array(data_dict[b"coarse_labels"], dtype=np.int64)  # shape (N,)

    # Convert data to images: reshape & transpose
    # Data is row-major, channels R, G, B, each 1024 entries
    # So each image: first 1024 = R channel, next 1024 = G, last 1024 = B
    # and each channel is 32x32, row-major
    num_samples = data.shape[0]
    # reshape to (N, 3, 32, 32)
    imgs = data.reshape((num_samples, 3, 32, 32))
    # transpose to (N, 32, 32, 3)
    imgs = imgs.transpose(0, 2, 3, 1)

    return imgs, fine, coarse


def load_meta(data_dir: str) -> Dict[str, List[str]]:
    """
    Load label names (fine and coarse) from meta file.

    Returns:
        A dict with keys:
          - 'fine_label_names': list of 100 names (strings)
          - 'coarse_label_names': list of 20 names
    """
    file_path = os.path.join(data_dir, "meta")
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    # Keys: b'fine_label_names', b'coarse_label_names'
    fine_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in data[b"fine_label_names"]]
    coarse_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in data[b"coarse_label_names"]]
    return {"fine_label_names": fine_names, "coarse_label_names": coarse_names}
