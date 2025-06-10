"""ThreadedTvioDataset: A multithreaded wrapper for VisionDataset.

This class wraps a map-style VisionDataset to make it iterable and
multithreaded in the fetch from storage. It uses a thread pool to
fetch samples in parallel, allowing for efficient data loading.
"""

import collections
import os
from copy import deepcopy
from multiprocessing.pool import AsyncResult, ThreadPool
from typing import TYPE_CHECKING, Any, Self

import torch
from torch import multiprocessing as mp
from torch.utils import data
from torchvision import datasets  # type: ignore
from torchvision.datasets.vision import StandardTransform  # type: ignore
from torchvision.io import ImageReadMode  # type: ignore
from torchvision.transforms import v2 as transforms  # type: ignore

from .shadow_dsets import ShadowBytesImageDataset

if TYPE_CHECKING:
    from .shadow_dsets import ShadowBytesImage

_DEFAULT_RNG_SEED = 20230829


class ThreadedTvioDataset(data.IterableDataset):
    """Wraps a map-style VisionDataset to be iterable -- and multithreaded in the fetch from storage."""

    def __init__(
        self,
        base_dset: datasets.VisionDataset,
        n_workers: int = 1,
        batch_size: int = 1,
        n_td_fetchers: int = 1,
        n_td_samples_prefetch: int = 1,
        *,
        shuffle: bool = False,
        rng_seed: int = _DEFAULT_RNG_SEED,
        image_decode_mode: ImageReadMode = ImageReadMode.RGB,
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
        self._decode_mode = image_decode_mode
        self.n_batches = self._base_dset_len // self._batch_size
        self._prev_pid = os.getpid()
        self._is_iter_called = False
        self._make_shared()

        self.n_td_samples_prefetch = n_td_samples_prefetch
        self.n_td_fetchers = n_td_fetchers
        self.first_time = True
        tr, ttr, trs = self._base_dset.transform, self._base_dset.target_transform, self._base_dset.transforms

        if trs is not None:
            self._transforms = trs
        elif tr is not None or ttr is not None:
            self._transforms = StandardTransform(tr, ttr)
        else:
            self._transforms = None
        self._base_dset.transform, self._base_dset.target_transform, self._base_dset.transforms = (
            None,
            None,
            None,
        )
        pil_handle_trf = StandardTransform(transforms.ToImage(), None)
        base_sample = pil_handle_trf(*self._base_dset[0])
        if self._transforms is not None:
            self._sample = self._transforms(*base_sample)
        else:
            self._sample = base_sample

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
        return self._sample

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
        self._td_scroll_idx = 0

        self._len_idxes_to_fetch = len(self._curr_wrkr_idxes_to_fetch)

        if self.first_time:
            self._shadow_dset = ShadowBytesImageDataset(self._base_dset)
            self._td_pool = ThreadPool(
                self.n_td_fetchers,
            )
        self._td_jobs: collections.deque[AsyncResult[list[tuple[ShadowBytesImage | int, ...]]]] = (
            collections.deque()
        )
        self._shadow_samples_lst: list[tuple[ShadowBytesImage | int, ...]] = []
        self._shadow_samples_lst_idx = 0
        self._is_iter_called = True

        for _ in range(2):
            self._scroll_add_job()

        return self

    def _scroll_add_job(self) -> None:
        if self._td_scroll_idx < self._len_idxes_to_fetch:
            idxes_to_fetch = self._curr_wrkr_idxes_to_fetch[
                self._td_scroll_idx : self._td_scroll_idx + self.n_td_samples_prefetch
            ]
            self._td_jobs.append(
                self._td_pool.map_async(
                    self._shadow_dset.__getitem__,
                    iterable=idxes_to_fetch,
                ),
            )
            self._td_scroll_idx += self.n_td_samples_prefetch

    def __next__(self) -> Any:
        if not self._is_iter_called:
            msg = "You must call __iter__() before __next__()"
            raise RuntimeError(msg)
        if self._shadow_samples_lst_idx >= len(self._shadow_samples_lst):
            if len(self._td_jobs) > 0:
                self._shadow_samples_lst = self._td_jobs.popleft().get()
                self._shadow_samples_lst_idx = 0
            else:
                self._is_iter_called = False
                raise StopIteration
            self._scroll_add_job()
        shadow_sample = self._shadow_samples_lst[self._shadow_samples_lst_idx]
        self._shadow_samples_lst_idx += 1
        true_sample = self._shadow_dset.sample_to_tensor(shadow_sample, mode=self._decode_mode)
        if self._transforms is not None:
            true_sample = self._transforms(*true_sample)
        return true_sample
