"""A manager for multiprocess H2D.

The classes given here include all the events and the dataclasses/contexts (appropriately bound)
that are needed for H2D. Further, the main class creates the background thread responsible for
H2D.

What these classes do NOT do is to create the worker processes. Nor is the necessary function
included. This is because that is highly dependent on the intent/ use case of the downstream user.

The necessary imperative code for every possible use case cannot be conjured, therefore, with the
exception of the background H2D thread that is the core, the classes below are effectively just
like dataclasses -- they hold attributes more than acting on them.

Note: 'Component' here relates to the entries in a single sample of a dataset.
E.g. if a single sample has an X tensor and a y tensor, that is two components.
THis disambiguation is necessary because we have splits, chunks, batches etc.
"""

import threading
from collections.abc import Sequence

import torch

from ..utils._safewait import safewait  # noqa: TID252
from . import _dcl_ctx


class H2DWrkrView:
    """Pass this to an individual worker."""

    def __init__(self, pinsh_sync_dcl: _dcl_ctx.PinshDcl) -> None:
        """Initialize the Worker View.

        Pass these to workers depending on your use case.

        Args:
            pinsh_sync_dcl (PinshDcl): The PinshDcl object to be manipulated by this worker.

        """
        self.pinsh_dcl = pinsh_sync_dcl

    def pinsh_write_ctx(self) -> _dcl_ctx.PinshWriteCtx:
        return _dcl_ctx.PinshWriteCtx(self.pinsh_dcl)

    def _pinsh_read_ctx(self) -> _dcl_ctx.PinshReadCtx:
        return _dcl_ctx.PinshReadCtx(self.pinsh_dcl)

    def reset(self) -> None:
        """Reset the worker view's events and flags."""
        self.pinsh_dcl.mpv_read_done.clear()
        self.pinsh_dcl.mpv_write_done.clear()
        self.pinsh_dcl.flag_resetted = True


class H2DManager:
    """Manage H2D transfers equivalent to a single component tensor."""

    def __init__(
        self,
        shapes: Sequence[tuple[int, ...]],
        dtypes: Sequence[torch.dtype],
        compute_stream: torch.cuda.Stream,
        n_host_batches: int = 2,
        n_cuda_batches: int = 2,
    ) -> None:
        """Initialize the H2D manager.

        Args:
            shapes (tuple[tuple[int, ...], ...]):
                The shapes of the tensors to be transferred.
            dtypes (tuple[torch.dtype, ...]):
                The dtypes of the tensors to be transferred.
            compute_stream (torch.cuda.Stream):
                The stream being used outside this class for computation.
                This is NOT the stream that will be used for the H2D transfer.
                This is the stream that will sync with the H2D transfer stream.
            n_host_batches (int, optional):
                The number of host batches. Defaults to 2.
                Should be equal to the number of worker processes.
            n_cuda_batches (int, optional):
                The number of CUDA batches. Defaults to 2.
                Ideally, can remain 2. This is just so that H2D happens smoothly
                without blocking compute.

        """
        self._n_host_batches = n_host_batches
        self._n_cuda_batches = n_cuda_batches
        self._pinsh_dcl_tup = tuple(
            _dcl_ctx.PinshDcl.fromscratch(shapes, dtypes) for _ in range(self._n_host_batches)
        )
        self._device = compute_stream.device
        self._custr_trns = torch.cuda.Stream(device=self._device)
        self._cutens_dcl_tup = tuple(
            _dcl_ctx.CuBufDcl.fromscratch(shapes, dtypes, compute_stream, self._custr_trns)
            for _ in range(self._n_cuda_batches)
        )

        self._i_pinsh_read = -1
        self._i_cu_write = -1
        self._i_cu_read = -1
        self._tdv_shutdown: threading.Event = threading.Event()
        self._tdv_pause: threading.Event = threading.Event()
        self._tdv_is_paused: threading.Event = threading.Event()
        self._tdv_resume: threading.Event = threading.Event()

        self._tdthread_h2d = threading.Thread(target=self._h2d_thread_fn, daemon=True)
        self._tdthread_h2d.start()

    def _pinsh_read_ctx(self) -> _dcl_ctx.PinshReadCtx:
        self._i_pinsh_read += 1
        to_return_idx = self._i_pinsh_read % self._n_host_batches
        to_return_dcls = self._pinsh_dcl_tup[to_return_idx]
        return _dcl_ctx.PinshReadCtx(to_return_dcls)

    def _cubuf_write_ctx(self, *, host_sync_on_exit: bool = False) -> _dcl_ctx.CuBufWriteCtx:
        self._i_cu_write += 1
        to_return_idx = self._i_cu_write % self._n_cuda_batches
        to_return_dcls = self._cutens_dcl_tup[to_return_idx]
        return _dcl_ctx.CuBufWriteCtx(to_return_dcls, host_sync_on_exit=host_sync_on_exit)

    def next_cubuf_read_ctx(self, *, host_sync_on_exit: bool = False) -> _dcl_ctx.CuBufReadCtx:
        self._i_cu_read += 1
        to_return_idx = self._i_cu_read % self._n_cuda_batches
        to_return_dcls = self._cutens_dcl_tup[to_return_idx]
        return _dcl_ctx.CuBufReadCtx(to_return_dcls, host_sync_on_exit=host_sync_on_exit)

    def _h2d_thread_fn(self) -> None:
        """Thread to handle the H2D transfer."""
        with torch.cuda.stream(self._custr_trns):  # type: ignore
            while not self._tdv_shutdown.is_set():
                if self._tdv_pause.is_set():
                    self._tdv_is_paused.set()
                    self._tdv_pause.clear()
                    safewait(self._tdv_resume)
                    self._tdv_resume.clear()

                with self._pinsh_read_ctx() as pinsh_read_ctx:  # noqa: SIM117
                    with self._cubuf_write_ctx(host_sync_on_exit=True) as cutens_write_ctx:
                        for pinsh_buf, cutens_buf in zip(pinsh_read_ctx.buffers, cutens_write_ctx.buffers):
                            if self._tdv_pause.is_set() or pinsh_read_ctx.is_dirty:
                                cutens_write_ctx.set_dirty()
                                break
                            pinsh_buf.copyto(cutens_buf, non_blocking=True)

    def get_wrkr_view(self, idx: int) -> H2DWrkrView:
        """Get the worker view for the SingleH2D context manager.

        Returns:
            WrkrView: The worker view.

        """
        return H2DWrkrView(self._pinsh_dcl_tup[idx])

    def reset(self) -> None:
        """To pause the h2d background thread and reset the events.

        It makes sense to call this at the start of a new epoch.
        Warning: this should be called only after ensuring none of the child processes
        are holding any context managers. Barrier synchronization events can be employed for the same.
        """
        self._tdv_pause.set()
        self._td_force_break_deadlocks()
        safewait(self._tdv_is_paused)
        self._tdv_is_paused.clear()

        self._i_pinsh_read, self._i_cu_write, self._i_cu_read = -1, -1, -1
        self._reset_all_dcls()
        self._tdv_resume.set()

    def _reset_all_dcls(self) -> None:
        """Reset all the data classes. To be run only when the H2D thread and all child workers are paused."""
        for idx in range(self._n_host_batches):
            self._pinsh_dcl_tup[idx].mpv_read_done.clear()
            self._pinsh_dcl_tup[idx].mpv_write_done.clear()
            self._pinsh_dcl_tup[idx].mpv_buf_dirty.clear()
            self._pinsh_dcl_tup[idx].flag_resetted = True
        for idx in range(self._n_cuda_batches):
            self._cutens_dcl_tup[idx].cuv_write_done.synchronize()
            self._cutens_dcl_tup[idx].cuv_read_done.synchronize()
            self._cutens_dcl_tup[idx].mpv_read_queued.clear()
            self._cutens_dcl_tup[idx].mpv_write_queued.clear()
            self._cutens_dcl_tup[idx].mpv_buf_dirty.clear()
            self._cutens_dcl_tup[idx].first_write = True

    def _td_force_break_deadlocks(self) -> None:
        """Force break any deadlocks.

        This is to be used for reset and shutdown, and again,
        requires the user to ensure that no child processes are holding any context managers.
        """
        for idx in range(self._n_host_batches):
            self._pinsh_dcl_tup[idx].mpv_read_done.clear()
            self._pinsh_dcl_tup[idx].mpv_write_done.set()

        for idx in range(self._n_cuda_batches):
            self._cutens_dcl_tup[idx].cuv_write_done.synchronize()
            self._cutens_dcl_tup[idx].mpv_write_queued.clear()
            self._cutens_dcl_tup[idx].mpv_read_queued.set()
            self._cutens_dcl_tup[idx].cuv_read_done.synchronize()

    def shutdown_h2d_thread(self) -> None:
        """Shutdown the H2D thread.

        Call only after all the worker processes have been terminated.
        Forcibly sets the events to ensure no deadlocks -- so will interfere with any
        child process workers still alive.
        """
        self._tdv_shutdown.set()
        self._tdv_pause.clear()
        self._tdv_is_paused.clear()
        self._tdv_resume.set()
        self._td_force_break_deadlocks()

    def __del__(self) -> None:
        """Shutdown the H2D thread.

        #TODO ? Is this necessary now that the thread is daemonic?
        """
        self.shutdown_h2d_thread()
