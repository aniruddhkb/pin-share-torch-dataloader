from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torchvision.datasets import VisionDataset  # type: ignore
from torchvision.io.image import ImageReadMode, decode_image  # type: ignore

if TYPE_CHECKING:
    import torchvision.transforms.v2 as T  # type: ignore  # noqa: N812
    from datasets import arrow_dataset  # type: ignore


class HFImageTorchDset(VisionDataset):
    """A PyTorch dataset wrapper for Hugging Face datasets containing images.

    This class is designed to work with datasets that have images in the first column
    and labels in the second column.
    """

    def __init__(
        self,
        hf_dset: arrow_dataset.Dataset,
        transform: T.Transform | None = None,
        *,
        decode_mode: ImageReadMode = ImageReadMode.UNCHANGED,
    ) -> None:
        """Initialize the HFImageTorchDset.

        Args:
            hf_dset:
              The Hugging Face dataset to load.
              It is assumed that the first column is the image and the second column is the label.
            transform:
                Optional torchvision.v2 Transform to be applied on the image.
                Must accept a torch.Tensor/torchvision.image.Image, without the target.
            decode_mode:
                The mode to use for decoding the image.
                See `torchvision.io.image.decode_image` for more details.
                Defaults to ImageReadMode.UNCHANGED.

        """
        self._hf_dset = hf_dset
        self._hf_dset.set_format(type="arrow")
        self._transform = transform
        self._decode_mode = decode_mode

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._hf_dset.num_rows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return a sample from the dataset.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A tuple containing the image and its label.

        """
        label: int = self._hf_dset[idx][1][0].as_py()

        img_bytes: bytes = self._hf_dset[idx][0][0][0].as_py()
        img_tensor_raw = torch.frombuffer(img_bytes, dtype=torch.uint8)
        img_tensor = decode_image(img_tensor_raw, mode=self._decode_mode)

        if self._transform is not None:
            return self._transform(img_tensor), label
        return img_tensor, label
