"""Take a VisionDataset and create a CachedVisionDataset where the tensors -- not the images -- are cached.

The assumptions:

1. The VisionDataset ("orig_dset") is a torchvision.datasets.VisionDataset with __getitem__ and __len__ .
2. orig_dset.__getitem__ returns a tuple[tensor, int]
3. Any transforms applied are deterministic -- they are not, say, random crops, flips or rotations.

What is created by the .make_cached_dset():

CachedVisionDataset:
A subclass of VisionDataset.
__init__ accepts a path to a directory where the tensors have been cached.
__getitem__ returns the tensor and label from the cache.
__len__ returns the number of tensors in the cache.

The cache has the following structure:
<cache_dir>/
    meta.csv
    tensors/
        0.pt
        1.pt
        2.pt
        ...

The meta.csv file has the following columns:

idx:int = The index of the tensor in the cache.
dtype:str = The data type of the tensor. For now, must be one of ["fp16","fp32"]
shape:str = The shape of the tensor. The expected format is d1-d2-...-dn with no spaces.
label:int = The label of the tensor. This is the label from the original dataset.

"""

import os

import numpy as np
import torch
from torchvision.datasets import VisionDataset  # type: ignore
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

STR_TO_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "uint8": np.uint8,
}


class CachedVisionDataset(VisionDataset):
    """A cached version of a VisionDataset."""

    @classmethod
    def _one_row_metadata(cls, orig_dset: VisionDataset, idx: int) -> tuple[str, str, str, str]:
        """Create a single row of metadata for the cached tensor.

        Args:
            orig_dset: The original VisionDataset.
            idx: The index of the tensor in the cache.

        Returns:
            A dictionary containing the metadata for the tensor.

        """
        tensor, label = orig_dset[idx]
        if not isinstance(tensor, torch.Tensor):
            msg = f"Expected tensor, got {type(tensor)}"
            raise TypeError(msg)
        if not isinstance(label, int):
            msg = f"Expected int, got {type(label)}"
            raise TypeError(msg)

        return (
            str(idx),
            str(tensor.dtype).split(".")[-1],
            "-".join([str(x) for x in tensor.shape]),
            str(label),
        )

    @classmethod
    def _make_metadata(cls, orig_dset: VisionDataset, cache_dir: str) -> None:
        with open(os.path.join(cache_dir, "meta.csv"), "w") as f:
            f.write("idx,dtype,shape,label\n")
            for idx in tqdm(range(len(orig_dset)), desc="Creating metadata", mininterval=5.0):
                row = cls._one_row_metadata(orig_dset, idx)
                f.write(",".join(row) + "\n")

    @classmethod
    def make_cached_dset(
        cls,
        orig_dset: VisionDataset,
        cache_dir: str,
    ) -> None:
        """Create a CachedVisionDataset from a VisionDataset.

        Args:
            orig_dset: The original VisionDataset to cache.
            cache_dir: The directory where the tensors will be cached.

        Returns:
            A CachedVisionDataset with the tensors cached in the specified directory.

        """
        os.makedirs(os.path.join(cache_dir, "tensors"), exist_ok=False)

        # Create the metadata file
        cls._make_metadata(orig_dset, cache_dir)

        # Cache the tensors
        tensor: torch.Tensor
        for idx in tqdm(range(len(orig_dset)), desc="Caching tensors", mininterval=5.0):
            tensor, _ = orig_dset[idx]
            tensor_bytes_path = os.path.join(cache_dir, "tensors", f"{idx}.bin")
            tensor_bytes = tensor.numpy().tobytes()
            with open(tensor_bytes_path, "wb") as f:
                f.write(tensor_bytes)

    def __init__(self, cache_dir: str) -> None:
        """Initialize the CachedVisionDataset.

        Args:
            cache_dir: The directory where the tensors are cached.

        """
        self._cache_dir = cache_dir
        self._meta = np.genfromtxt(
            os.path.join(self._cache_dir, "meta.csv"),
            delimiter=",",
            dtype=str,
        ).tolist()[1:]
        self._len = len(self._meta)
        for i in tqdm(range(self._len), desc="Loading metadata", mininterval=5.0):
            curr = self._meta[i]
            curr[0] = int(curr[0])
            assert curr[0] == i
            curr[1] = STR_TO_DTYPE[curr[1]]
            curr[2] = [int(j) for j in curr[2].split("-")]
            curr[3] = int(curr[3])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return a sample from the dataset.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A tuple containing the tensor and its label.

        """
        if idx >= self._len or idx < 0:
            msg = f"Index {idx} out of bounds for dataset of length {self._len}"
            raise IndexError(msg)

        tensor_bytes_path = os.path.join(self._cache_dir, "tensors", f"{idx}.bin")
        with open(tensor_bytes_path, "rb") as f:
            tensor_bytes = f.read()

        meta_row = self._meta[idx]
        np_dtype, shape, label = meta_row[1:]
        nd_arr = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape)

        tens = torch.from_numpy(nd_arr)

        return tens, label

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            The number of samples in the dataset.

        """
        return self._len
