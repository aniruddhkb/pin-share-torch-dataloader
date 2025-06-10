import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class GetMeanVarianceStd:
    """To get the mean and variance of each channel of an image dataset.

    For normalizing the dataset before training or inference.
    """

    def __init__(self, dataset: Dataset) -> None:
        """Initialize the GetMeanVariance class.

        Args:
            dataset: The dataset to compute the mean and variance for.
                     It is assumed that the first column is a torch.Tensor,
                     in channels_first format.

        """
        if not isinstance(dataset, Dataset):
            msg = "dataset must be a torch.utils.data.Dataset"
            raise TypeError(msg)

        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            msg = "dataset must be a torch.utils.data.Dataset with __len__ and __getitem__ methods"
            raise TypeError(msg)

        if not isinstance(dataset[0], tuple):
            msg = "dataset must return a tuple of (image, label)"
            raise TypeError(msg)

        if not isinstance(dataset[0][0], torch.Tensor):
            msg = "dataset must return a torch.Tensor as the first element of the tuple"
            raise TypeError(msg)

        if dataset[0][0].dim() != 3:  # noqa: PLR2004
            msg = "dataset must return a 3D torch.Tensor as the first element of the tuple"
            raise TypeError(msg)

        self._dataset = dataset
        example_tens: torch.Tensor = dataset[0][0]

        self._shape = example_tens.shape

        self._n_channels: int = self._shape[0]

        self._len = dataset.__len__()  # type: ignore
        self._pixels_per_channel: int = self._shape[1] * self._shape[2]
        self._n = self._len * self._pixels_per_channel

        self._sum_x = torch.zeros(self._n_channels, dtype=torch.float32)
        self._sum_x2 = torch.zeros(self._n_channels, dtype=torch.float32)

        for i in tqdm(range(self._len)):
            x: torch.Tensor = dataset[i][0]
            if x.shape != self._shape:
                msg = f"dataset must return a tensor of shape {self._shape} as the first element of the tuple"
                raise ValueError(msg)
            self._sum_x += self._sum_preserving_channels(x)
            self._sum_x2 += self._sum_preserving_channels(x**2)
        self.mean = self._sum_x / self._n
        self.variance = (self._sum_x2 / self._n) - (self.mean**2)
        self.std = torch.sqrt(self.variance)

    def _sum_preserving_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the channels of a tensor preserving the channel dimension.

        Args:
            x: The input tensor to sum.

        """
        if x.dim() != 3:  # noqa: PLR2004
            msg = "x must be a 3D tensor"
            raise ValueError(msg)

        return torch.sum(x, dim=(1, 2), keepdim=False)
