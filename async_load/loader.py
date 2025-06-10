"""A dataloader that directly places the batch on the device.

This is the main entry point for users.

Note: After the post_fetch_transform, all components are expected to either be tensors or ints.
The default int type is taken to be int64.

"""

from __future__ import annotations

import time

from torch import multiprocessing as mp

from .file2mem.threaded_tvio_dset import ThreadedTvioDataset
from .utils._prflrs import get_profiler
from .utils._safewait import safewait

if mp.get_start_method(allow_none=True) != "spawn":
    msg = f"Multiprocessing start method is not correctly set.\
        Expected 'spawn', got '{mp.get_start_method(allow_none=True)}'."

import os
import signal
from typing import TYPE_CHECKING, Self, TypedDict

import torch
from torch.utils.data import IterableDataset

from .h2d._mgr import H2DManager, H2DWrkrView
from .utils._iter_wrap import IterWrapDataset
from .utils._pin_shr_mem import disable_cudart_error

if TYPE_CHECKING:
    from multiprocessing import synchronize

    from torch.utils.data import Dataset

    from async_load.h2d._dcl_ctx import CuBufReadCtx

_DEFAULT_RNG_SEED = 29082023
_WRKR_ID = 0
_STOP_ITERATION_SIGNAL: int = signal.SIGUSR1
_WRKR_ITERATION_STOPPED = True

disable_cudart_error()


class _WorkerArgs(TypedDict):
    """Arguments for the worker process."""

    dset: IterableDataset
    batch_size: int
    worker_id: int
    h2d_wrkr_view: H2DWrkrView
    mpbarr_a: synchronize.Barrier
    mpbarr_b: synchronize.Barrier
    torch_num_threads: int
    profile_trace: bool
    profile_trace_path: str
    profile_wait: int
    profile_warmup: int
    profile_record: int


class CudaDataLoader:
    """DataLoader that directly sends to CUDA.

    This class is a wrapper around the ThreadedFetchDataset and H2DManager classes.
    DOES NOT FOLLOW PYTORCH DATALOADER SEMANTICS COMPLETELY.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        base_dset: Dataset | IterableDataset,
        batch_size: int,
        n_wrkrs: int,
        shuffle: bool = False,
        n_cubufs: int = 2,
        torch_num_threads: int = 1,
        cu_stream: torch.cuda.Stream | None = None,
        rng_seed: int = _DEFAULT_RNG_SEED,
        profile_trace: bool = False,
        profile_trace_path: str = "",
        profile_wait: int = 0,
        profile_warmup: int = 0,
        profile_record: int = 0,
    ) -> None:
        """Initialize the CudaDataLoader."""
        # Arg validation and setting.
        dset = self._get_check_wrapped_dset(
            base_dset,
            n_wrkrs,
            batch_size,
            shuffle=shuffle,
            rng_seed=rng_seed,
        )

        self._init_make_flags_syncs_consts(n_wrkrs)

        self._h2d_mgr = self._init_get_h2d_manager(dset, batch_size, cu_stream, n_wrkrs, n_cubufs)
        self._mpproc_lst, self._dontuse = self._init_get_procs(
            dset=dset,
            batch_size=batch_size,
            h2d_manager=self._h2d_mgr,
            n_wrkrs=n_wrkrs,
            mpbarr_a=self._mpbarr_a,
            mpbarr_b=self._mpbarr_b,
            torch_num_threads=torch_num_threads,
            profile_trace=profile_trace,
            profile_trace_path=profile_trace_path,
            profile_wait=profile_wait,
            profile_warmup=profile_warmup,
            profile_record=profile_record,
        )
        self._mppidfd_lst = self._init_get_mppidfd_lst(self._mpproc_lst)

    def _get_check_wrapped_dset(
        self,
        dset: Dataset,
        n_wrkrs: int,
        batch_size: int,
        *,
        shuffle: bool,
        rng_seed: int,
    ) -> IterWrapDataset | IterableDataset:
        if not isinstance(dset, IterableDataset):
            to_return = IterWrapDataset(
                dset,
                n_workers=n_wrkrs,
                batch_size=batch_size,
                shuffle=shuffle,
                rng_seed=rng_seed,
            )
            self._len: int = to_return.n_batches
            return to_return
        self._len = int(float("inf"))
        return dset

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self._len

    def _init_make_flags_syncs_consts(self, n_wrkrs: int) -> None:  # Rework
        self._mpbarr_a = mp.Barrier(n_wrkrs + 1)
        self._mpbarr_b = mp.Barrier(n_wrkrs + 1)

        self._onceflag_not_called_iter = True

    def _init_get_procs(  # noqa: PLR0913
        self,
        *,
        dset: IterableDataset,
        batch_size: int,
        h2d_manager: H2DManager,
        n_wrkrs: int,
        mpbarr_a: synchronize.Barrier,
        mpbarr_b: synchronize.Barrier,
        torch_num_threads: int = 1,
        profile_trace: bool = False,
        profile_trace_path: str = "",
        profile_wait: int = 0,
        profile_warmup: int = 0,
        profile_record: int = 0,
    ) -> tuple[list[mp.Process], list[_WorkerArgs]]:
        wrkr_args_lst = [
            _WorkerArgs(
                dset=dset,
                batch_size=batch_size,
                worker_id=idx,
                h2d_wrkr_view=h2d_manager.get_wrkr_view(idx),
                mpbarr_a=mpbarr_a,
                mpbarr_b=mpbarr_b,
                torch_num_threads=torch_num_threads,
                profile_trace=profile_trace,
                profile_trace_path=profile_trace_path,
                profile_wait=profile_wait,
                profile_warmup=profile_warmup,
                profile_record=profile_record,
            )
            for idx in range(n_wrkrs)
        ]

        mpproc_lst = [
            (mp.Process(target=self._worker_fn, kwargs=wrkrargs, daemon=False)) for wrkrargs in wrkr_args_lst
        ]

        for mp_proc in mpproc_lst:
            mp_proc.start()
        return mpproc_lst, wrkr_args_lst

    def _init_get_mppidfd_lst(self, mpproc_lst: list[mp.Process]) -> list[int]:  # Freeze
        mppidfd_lst: list[int] = []
        for mp_proc in mpproc_lst:
            proc_pid = mp_proc.pid
            if proc_pid is None:
                msg = "Spawned process has no PID."
                raise RuntimeError(msg)
            proc_fd = os.pidfd_open(proc_pid, 0)
            if proc_fd == -1:
                msg = f"Failed to get pidfd for process {proc_pid}."
                raise RuntimeError(msg)
            mppidfd_lst.append(proc_fd)
        return mppidfd_lst

    def _init_get_h2d_manager(
        self,
        base_dset: IterableDataset,
        batch_size: int,
        compute_stream: torch.cuda.Stream | None,
        num_worker_processes: int,
        n_cuda_buffers: int,
    ) -> H2DManager:
        """Get a new H2D manager."""
        batch_shapes, dtypes = self._init_get_batch_shapes_dtypes(base_dset, batch_size)

        # Create the H2D manager.
        return H2DManager(
            shapes=batch_shapes,
            dtypes=dtypes,
            compute_stream=torch.cuda.current_stream() if compute_stream is None else compute_stream,
            n_host_batches=num_worker_processes,
            n_cuda_batches=n_cuda_buffers,
        )

    def _init_get_batch_shapes_dtypes(
        self,
        base_dset: IterableDataset,
        batch_size: int,
    ) -> tuple[list[tuple[int, ...]], list[torch.dtype]]:
        if isinstance(base_dset, (IterWrapDataset, ThreadedTvioDataset)):
            sample = base_dset.sample
        else:
            sample = base_dset.__iter__().__next__()

        if not isinstance(sample, (list, tuple)):
            sample = (sample,)
        shapes: list[list[int]] = []
        dtypes: list[torch.dtype] = []
        for component in sample:
            if isinstance(component, torch.Tensor):
                tensorcomponent: torch.Tensor = component  # for linting.
                shapes.append(list(tensorcomponent.shape))
                dtypes.append(tensorcomponent.dtype)
            elif isinstance(component, int):
                shapes.append([])
                dtypes.append(torch.int64)
            else:
                msg = f"Unsupported type {type(component)} in sample."
                raise TypeError(msg)

        return [(batch_size, *shape) for shape in shapes], dtypes

    def __iter__(self) -> Self:
        """To start an epoch.

        Must be called before every epoch.
        """
        if self._onceflag_not_called_iter:
            self._onceflag_not_called_iter = False

        else:
            for pidfd in self._mppidfd_lst:
                signal.pidfd_send_signal(pidfd, _STOP_ITERATION_SIGNAL, None, 0)

            self._mpbarr_a.wait()
            self._mpbarr_a.reset()
            self._h2d_mgr.reset()  # Reset the H2D manager -- call only after all workers are paused.
            self._mpbarr_b.wait()  # Wait for all workers to pause.
            self._mpbarr_b.reset()

        self._mpbarr_a.wait()
        self._mpbarr_a.reset()
        self._first_batch_of_epoch = True
        return self

    def __next__(self) -> CuBufReadCtx:
        """Get the next batch context.

        Returns:
            CudaBatchReadCtx: The next batch context.

            Wrap all code that requires the current value of the batch to be maintained,
            in this context. This will ensure that the relevant CUDA and mp events are
            waited for, set and cleared.

        Raises:
            StopIteration: If the iterator has been exhausted.
            RuntimeError: If the iterator has not been started.


        """
        if self._onceflag_not_called_iter:
            msg = "The iterator has not been started. Call __iter__() first."
            raise RuntimeError(msg)

        if self._first_batch_of_epoch:
            safewait(self._h2d_mgr._cutens_dcl_tup[-1].mpv_write_queued)  # noqa: SLF001
            self._first_batch_of_epoch = False

        to_return = self._h2d_mgr.next_cubuf_read_ctx()
        to_return.pre_wait()
        if to_return.is_dirty:
            raise StopIteration

        return to_return

    @classmethod
    def _worker_fn(  # noqa: C901, PLR0912, PLR0913, PLR0915
        cls,
        *,
        dset: IterableDataset,
        batch_size: int,
        worker_id: int,
        h2d_wrkr_view: H2DWrkrView,
        mpbarr_a: synchronize.Barrier,
        mpbarr_b: synchronize.Barrier,
        torch_num_threads: int = 1,
        profile_trace: bool = False,
        profile_trace_path: str = "",
        profile_wait: int = 0,
        profile_warmup: int = 0,
        profile_record: int = 0,
    ) -> None:
        """Worker function for the worker process."""

        def _usrsig_handler(*args, **kwargs) -> None:  # noqa: ANN002, ANN003, ARG001
            """Signal handler for SIGUSR1 aka StopIteration."""
            global _WRKR_ITERATION_STOPPED  # noqa: PLW0603
            if not _WRKR_ITERATION_STOPPED:
                _WRKR_ITERATION_STOPPED = True
                raise StopIteration

        signal.signal(_STOP_ITERATION_SIGNAL, _usrsig_handler)

        from .utils._dms import DMS

        _dms = DMS()
        torch.set_num_threads(torch_num_threads)
        with get_profiler(
            "torch_no_cuda",
            profile_wait,
            profile_warmup,
            profile_record,
            log_dir=profile_trace_path,
            bypassed=not profile_trace,
        ) as prof:
            try:
                if isinstance(dset, (IterWrapDataset, ThreadedTvioDataset)):
                    dset.set_worker_info(worker_id)
                global _WRKR_ID  # noqa: PLW0603
                _WRKR_ID = worker_id
                global _WRKR_ITERATION_STOPPED  # noqa: PLW0603
                row_views_lsts = h2d_wrkr_view.pinsh_dcl.row_views
                while True:  # epochs
                    dset_iter = dset.__iter__()
                    epoch_first_batch = True
                    h2d_wrkr_view.reset()

                    while True:  # Batches
                        if _WRKR_ITERATION_STOPPED and not epoch_first_batch:
                            break

                        try:
                            with h2d_wrkr_view.pinsh_write_ctx():
                                for row_idx in range(batch_size):
                                    try:
                                        sample = dset_iter.__next__()
                                    except StopIteration:
                                        _WRKR_ITERATION_STOPPED = True
                                    if _WRKR_ITERATION_STOPPED and not epoch_first_batch:
                                        h2d_wrkr_view.pinsh_dcl.mpv_buf_dirty.set()
                                        break
                                    for sample_comp, pinsh_comp in zip(sample, row_views_lsts[row_idx]):
                                        if isinstance(sample_comp, torch.Tensor):
                                            pinsh_comp.copy_(sample_comp)
                                        elif isinstance(sample_comp, int):
                                            pinsh_comp.fill_(sample_comp)
                                if epoch_first_batch:
                                    epoch_first_batch = False
                                    mpbarr_a.wait()
                                    _WRKR_ITERATION_STOPPED = False
                            prof.nextstep()
                        except StopIteration:
                            _WRKR_ITERATION_STOPPED = True
                            break
                    mpbarr_a.wait()
                    mpbarr_b.wait()
                    epoch_first_batch = True

            finally:
                prof.nextstep()
                print(f"Worker {worker_id} exiting.")
                # del h2d_wrkr_view

    def __del__(self) -> None:
        """Destructor for the CudaDataLoader."""
        for mp_proc in self._mpproc_lst:
            mp_proc.terminate()
        time.sleep(3)

        for mp_proc in self._mpproc_lst:
            if mp_proc.is_alive():
                mp_proc.kill()

        for mp_proc in self._mpproc_lst:
            mp_proc.join()
        self._h2d_mgr.shutdown_h2d_thread()
