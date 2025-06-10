"""Utilities for shadowing PIL Image class to return bytes instead of images."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from typing import IO, Self

import torch
from PIL import Image
from torch.utils import data
from torchvision.datasets import VisionDataset  # type: ignore
from torchvision.io import ImageReadMode, decode_image  # type: ignore

ORIG_OPEN = Image.open
ORIG_IMAGE = Image.Image


class _ShadowPathImage(Image.Image):
    """A shadow image class that can be used to shadow the PIL Image class, to return the img path."""

    def __init__(self, img_path: str) -> None:
        super().__init__()
        self.shadow_img_path = img_path

    @classmethod
    def shadow_path_open(cls, fp: str, *args, **kwargs) -> Self:  # noqa: ANN002, ANN003, ARG003
        """To NOT open the image and instead return the _ShadowPathImage."""
        return cls(fp)

    def convert(self, *args, **kwargs) -> Self:  # noqa: ANN002, ANN003, ARG002
        """To bypass super().convert which expects a PIL image."""
        return self


class _ShadowPathImageDataset(data.Dataset):
    """A dataset that returns the image path instead of the image."""

    def __init__(self, base_dset: VisionDataset) -> None:
        """To initialize the ShadowPathImageDataset.

        Args:
            base_dset (VisionDataset): The base dataset to shadow.

        """
        if not isinstance(base_dset, VisionDataset):
            msg = f"Expected base_dset to be of type 'VisionDataset', got {type(base_dset)}"
            raise TypeError(msg)
        self.base_dset = base_dset
        self.orig_open = Image.open
        self.orig_image = Image.Image
        ex_sample = self.base_dset[0]

        if not isinstance(ex_sample, Sequence):
            ex_sample = (ex_sample,)
        for ex_part in ex_sample:
            if not any(
                [
                    isinstance(ex_part, Image.Image),
                    isinstance(ex_part, int),
                ],
            ):
                msg = f"All parts of the sample should be of type PIL Image, int.\
                    Currently, it is {[type(prt) for prt in ex_sample]}.\
                    This can happen if transforms were applied to the dataset,\
                    or if it has additional non-PIL-image data that is in array or object form."
                raise TypeError(msg)
        if "loader" in dir(self.base_dset):
            self.base_dset.loader = self.loader  # type: ignore

    def __len__(self) -> int:
        """To return the length of the base dataset."""
        return len(self.base_dset)

    def __getitem__(self, idx: int) -> tuple[int | str]:
        """To return the _ShadowPathImage sample from the base dataset, along with the target if any."""
        try:
            Image.open = _ShadowPathImage.shadow_path_open  # type: ignore
            sample = self.base_dset[idx]
        finally:
            Image.open = self.orig_open
        return sample


class ShadowBytesImage(Image.Image):
    """Shadows the PIL Image class -- returns bytes instead of decoded image."""

    def __init__(self, bytesobj: bytes | None = None) -> None:
        super().__init__()
        if bytesobj is None:
            msg = """Expected bytesobj to be of type 'bytes', got 'None'.
Even with a valid dset, this can happen if the dset prefetches the whole dataset in __init__.
This is the behaviour of CIFAR10, CIFAR100, and MNIST. For these, use the regular dset,
instead of the shadow dset."""
            raise TypeError(msg)
        self.bytesobj = bytesobj

    @classmethod
    def shadow_open(cls, fp: str | IO[bytes] | bytes, *args, **kwargs) -> Self:  # noqa: ANN002, ANN003, ARG003
        """To open the image and return the ShadowImage."""
        if isinstance(fp, str):
            with open(fp, "rb") as f:  # noqa: PTH123
                return cls(f.read())
        elif isinstance(fp, BytesIO):
            return cls(fp.read())
        elif isinstance(fp, bytes):
            return cls(fp)
        else:
            msg = f"Expected fp to be of type 'str', 'bytes', or 'io.IOBase', got {type(fp)}"
            raise TypeError(msg)

    def convert(self, *args, **kwargs) -> Self:  # noqa: ANN002, ANN003, ARG002
        """To bypass super().convert which expects a PIL image."""
        return self

    @classmethod
    def shadow_loader(cls, file_path_or_bytes: str | bytes) -> Self:
        """To load the image as a ShadowImage."""
        if isinstance(file_path_or_bytes, bytes):
            return cls(file_path_or_bytes)
        elif isinstance(file_path_or_bytes, str):  # noqa: RET505
            with open(file_path_or_bytes, "rb") as f:  # noqa: PTH123
                return cls(f.read())
        else:
            msg = (
                f"Expected file_path_or_bytes to be of type 'str' or 'bytes', got {type(file_path_or_bytes)}"
            )
            raise TypeError(msg)

    def as_bytes(self) -> bytes:
        """To return the bytes of the image."""
        return self.bytesobj


class ShadowBytesImageDataset(data.Dataset):
    """A dataset that returns bytes ShadowImages instead of PIL Images."""

    def __init__(self, base_dset: VisionDataset) -> None:
        self.orig_open = Image.open
        self.orig_image = Image.Image
        self.base_dset = base_dset
        ex_sample = self.base_dset[0]
        if not isinstance(ex_sample, Sequence):
            ex_sample = (ex_sample,)
        for ex_part in ex_sample:
            if not any(
                [
                    isinstance(ex_part, Image.Image),
                    isinstance(ex_part, int),
                ],
            ):
                msg = f"All parts of the sample should be of type PIL Image, int.\
                    Currently, it is {[type(prt) for prt in ex_sample]}.\
                    This can happen if transforms were applied to the dataset,\
                    or if it has additional non-PIL-image data that is in array or object form."
                raise TypeError(msg)
        if "loader" in dir(self.base_dset):
            self.base_dset.loader = ShadowBytesImage.shadow_loader  # type: ignore

    def __len__(self) -> int:
        """To return the length of the base dataset."""
        return len(self.base_dset)

    def __getitem__(self, idx: int) -> tuple[int | ShadowBytesImage]:
        """To return the bytes sample from the base dataset."""
        try:
            Image.open = ShadowBytesImage.shadow_open  # type: ignore
            sample = self.base_dset[idx]
        finally:
            Image.open = self.orig_open
        return sample

    def get_bytes_sample(self, idx: int) -> tuple[bytes | int, ...]:
        """To return the bytes sample from the base dataset."""
        return self.sample_to_bytes(self.__getitem__(idx))

    @classmethod
    def sample_to_bytes(
        cls,
        sample: tuple[int | ShadowBytesImage, ...],
    ) -> tuple[bytes | int, ...]:
        """To return the bytes sample from the base dataset."""
        to_return: list[int | bytes] = []
        for part in sample:
            if isinstance(part, ShadowBytesImage):
                to_return.append(part.as_bytes())
            else:
                to_return.append(part)
        return tuple(to_return)

    @classmethod
    def sample_to_tensor(
        cls,
        sample: tuple[int | ShadowBytesImage, ...],
        mode: ImageReadMode = ImageReadMode.RGB,
    ) -> tuple[int | torch.Tensor, ...]:
        """To return the bytes sample from the base dataset."""
        to_return: list[int | torch.Tensor] = []
        for part in sample:
            if isinstance(part, ShadowBytesImage):
                to_return.append(
                    decode_image(torch.frombuffer(part.as_bytes(), dtype=torch.uint8), mode=mode)
                )
            else:
                to_return.append(part)
        return tuple(to_return)
