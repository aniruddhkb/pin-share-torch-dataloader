from typing import Any

from torch.utils import data


class TruncatedDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, num_samples: int) -> None:
        self._dataset = dataset
        self._len = min(num_samples, len(dataset))  # type: ignore

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self._dataset[index]
