from torch.utils import data


class RepeatedDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, num_repeats: int) -> None:
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self) -> int:
        return len(self.dataset) * self.num_repeats  # type: ignore

    def __getitem__(self, index: int) -> tuple:
        index = index % len(self.dataset)  # type: ignore
        return self.dataset[index]
