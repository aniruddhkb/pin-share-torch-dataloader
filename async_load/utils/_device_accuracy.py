"""Calculate accuracy metrics on the device."""

from __future__ import annotations

import torch


class CategoricalAccuracy:
    """To compute the typical categorical accuracy in classification tasks.

    At each batch, takes the y (ground truth) and yhat (predictions), and then at the end of an epoch,
    computes the accuracy as the number of correct predictions divided by the total number of predictions.

    Macro-micro will be added much later. As will top-1 vs top-K accuracy.
    """

    def __init__(
        self,
        n_classes: int,
        batch_size: int,
        max_n_batches: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cuda",
    ) -> None:
        """Initialize the DeviceAccuracy class.

        A lazy approach is decided against, for fear it may interfere with torch.compile .

        Args:
            n_classes (int): The number of classes for the predictions.
            batch_size (int): The batch size of the predictions.
            max_n_batches (int): The max no. of batches in an epoch.
                Set this to len(dataset)//batch_size if possible.
            dtype (torch.dtype): The data type of the predictions.
            device (torch.device | str): The device to use.
                If a string, it should be "cpu" or "cuda". If a torch.device, it should be a valid device.

        """
        self._max_n_samples = max_n_batches * batch_size
        self._n_views = max_n_batches
        self._batch_size = batch_size

        self._yhats_logits_tens = torch.zeros(
            (self._max_n_samples, n_classes),
            dtype=dtype,
            device=device,
        )

        self._y_s_tens = torch.zeros(
            (self._max_n_samples,),
            dtype=torch.int64,
            device=device,
        )

        self._yhats_ints_tens = torch.zeros_like(self._y_s_tens)

        self._yhats_logits_tens_views: list[torch.Tensor] = [
            self._yhats_logits_tens[i * batch_size : (i + 1) * batch_size] for i in range(self._n_views)
        ]

        self._y_s_tens_views: list[torch.Tensor] = [
            self._y_s_tens[i * batch_size : (i + 1) * batch_size] for i in range(self._n_views)
        ]

    def reset(self) -> None:
        """Reset the accuracy object."""
        self._yhats_logits_tens.zero_()
        self._y_s_tens.zero_()
        self._yhats_ints_tens.zero_()
        self._scroll_idx = 0

    def __call__(
        self,
        yhats_logits: torch.Tensor,
        y_s: torch.Tensor,
    ) -> None:
        """Compute the accuracy.

        WARNING: This function does not perform any validations so as to not accidentally trigger a CUDA sync.

        HEED THIS WELL.

        It is not this function that will throw an error, but the CUDA Runtime itself.
        Good luck debugging that.

        No actual computation happens here. This just takes and stores the data.

        The actual computation happens in the calculate function.
        The reason for this is to avoid any CUDA syncs, which can be expensive.

        Args:
            y_s (torch.Tensor): The ground truth.
                Expected to be of shape (batch_size,) and of type torch.int64.
            yhats_logits (torch.Tensor):
                The predictions. Expected to be of shape (batch_size, n_classes), type as passed in __init__.

        """
        self._yhats_logits_tens_views[self._scroll_idx].add_(yhats_logits)
        self._y_s_tens_views[self._scroll_idx].add_(y_s)
        self._scroll_idx += 1

    def compute(self) -> float:
        """Calculate the accuracy.

        Unlike the prior functions this absolutely _will_ trigger a CUDA sync.
        """
        total = self._batch_size * self._scroll_idx
        if total == 0:
            return 0.0
        torch.argmax(self._yhats_logits_tens, dim=1, keepdim=False, out=self._yhats_ints_tens)
        torch.sub(self._y_s_tens, self._yhats_ints_tens, out=self._y_s_tens)
        incorrect = torch.count_nonzero(self._y_s_tens).item()
        return 1 - (incorrect / total)
