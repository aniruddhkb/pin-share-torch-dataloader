"""Make an iterable dataset from a map-style dataset, while handling indices across workers."""

import os
from copy import deepcopy
from typing import Any, Self

import torch
from torch import multiprocessing as mp
from torch.utils import data

_DEFAULT_RNG_SEED = 42


class IterWrapDataset(data.IterableDataset):
    """Wraps a map-style dataset to be iterable.

    Also handles the randomization and sampling of the dataset.
    This is a very basic implementation that merely supports
    shuffled indices.
    """

    def __init__(
        self,
        base_dset: data.Dataset,
        n_workers: int = 1,
        batch_size: int = 1,
        *,
        shuffle: bool = False,
        rng_seed: int = _DEFAULT_RNG_SEED,
    ) -> None:
        if not all(i in dir(base_dset) for i in ("__len__", "__getitem__")):
            msg = "base_dset must be a map-style dataset having __len__ and __getitem__ methods."
            raise ValueError(msg)

        self._base_dset = base_dset
        self._shuffle = shuffle
        self._orig_rng_seed = rng_seed
        self._rng_seed = rng_seed
        self._wrkr_id = 0
        self._n_wrkrs = n_workers
        self._batch_size = batch_size
        self._base_dset_len = len(self._base_dset)  # type: ignore
        self.n_batches = self._base_dset_len // self._batch_size
        self._prev_pid = os.getpid()
        self._is_iter_called = False
        self._make_shared()

    def _make_shared(self) -> None:
        self._iter_entry_barr = mp.Barrier(self._n_wrkrs)
        self._iter_exit_barr = mp.Barrier(self._n_wrkrs)
        self._shared_idxes: torch.Tensor = torch.zeros(self._base_dset_len, dtype=torch.int64).share_memory_()
        self._shared_views = [
            self._shared_idxes[i * self._batch_size : (i + 1) * self._batch_size]
            for i in range(self.n_batches)
        ]

    def __len__(self) -> int:
        """Return the length of the base dataset."""
        return self._base_dset_len

    def _populate_shared_idxes(self) -> None:
        """Get the next indices to be fetched from the dataset.

        This is used to split the dataset among workers.
        """
        if self._shuffle:
            torch.randperm(
                self._base_dset_len,
                generator=torch.Generator().manual_seed(self._rng_seed),
                out=self._shared_idxes,
            )
        else:
            torch.arange(end=self._base_dset_len, out=self._shared_idxes)

    def set_worker_info(self, wrkr_id: int) -> None:
        """Set the worker id and number of workers."""
        new_pid = os.getpid()
        if new_pid != self._prev_pid:
            self._prev_pid = new_pid
            self._rng_seed = self._orig_rng_seed
            self._base_dset = deepcopy(self._base_dset)
            self._is_iter_called = False
        self._wrkr_id = wrkr_id

    @property
    def sample(self) -> Any:  # noqa: ANN401
        return self._base_dset[0]

    def __iter__(self) -> Self:
        """Initialize the bells and whistles needed to make this work."""
        self._iter_entry_barr.wait()
        self._iter_entry_barr.reset()
        if self._wrkr_id == 0:
            self._rng_seed += 1
            self._populate_shared_idxes()
        self._iter_exit_barr.wait()
        self._iter_exit_barr.reset()
        self._curr_wrkr_views = self._shared_views[self._wrkr_id :: self._n_wrkrs]
        self._curr_wrkr_idxes_to_fetch: list[int] = torch.cat(self._curr_wrkr_views).tolist()
        self._next_idx = 0
        self._len_idxes_to_fetch = len(self._curr_wrkr_idxes_to_fetch)
        self._is_iter_called = True
        return self

    def __next__(self) -> Any:  # noqa: ANN401
        if not self._is_iter_called:
            msg = "You must call __iter__() before __next__()"
            raise RuntimeError(msg)
        if self._next_idx >= self._len_idxes_to_fetch:
            self._is_iter_called = False
            raise StopIteration
        idx = self._curr_wrkr_idxes_to_fetch[self._next_idx]
        self._next_idx += 1
        return self._base_dset[idx]
