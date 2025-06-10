"""Contains dataclasses for the H2D transfer and synchronization events.

DATACLASSES

The dataclasses include:

1. A PinshBuffer or a CUDA tensor.
2. Python multiprocessing events (mpv) to indicate the completion of a read or write.
3. For the CUDA dataclass, corresponding CUDA events (cuv) .
4. An event to indicate whether the buffer is "dirty".
This is useful if, say, the worker runs out of data and must stop and release the buffer
-- but the main process must not use it for , say, training.

So each buffer has with it, all the events needed for both worker and main to handle the data transfer.

One quirk to note: for CUDA transfers, there are both mpv and cuv .

This is because, a cuv is not really an event -- it's a marker placed in a CUDA stream.
If that cuv is not yet placed in the stream (because, say, the host was busy), then
a call to <cuv>.synchronize() will go through -- even if the thing it is to wait for is
not even initiated!

So what is done is to first place the <cuv>.record , then call <mpv>.set .
On the waiting end, do <mpv>.wait and then <cuv>.synchronize or <cuv>.wait as applicable.

That way, when the CUDA wait/synchronize is called, you are sure that the event was placed on the stream.

CONTEXTS

The contexts enable the events to be set/waited/synchronized/cleared without involving
the user of this API. All the necessary wait operations are packaged in __enter__
and all the set operations in __exit__ , with the API user being able to put
their code between these two, since it's a context.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import torch
import torch.multiprocessing as mp

from async_load.utils._pin_shr_mem import PinshBuffer
from async_load.utils._safewait import safewait

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing import synchronize

    from torch._C import _CudaEventBase, _CudaStreamBase  # type: ignore


@dataclass
class PinshDcl:
    """Dataclass for multi-component H2D transfer."""

    buffers: tuple[PinshBuffer, ...]
    mpv_read_done: synchronize.Event
    mpv_write_done: synchronize.Event
    mpv_buf_dirty: synchronize.Event
    row_views: tuple[tuple[torch.Tensor, ...], ...]
    flag_resetted: bool = True

    @classmethod
    def fromscratch(cls, shapes: Sequence[tuple[int, ...]], dtypes: Sequence[torch.dtype]) -> Self:
        """Create a MultiPinshDcl object from scratch."""
        pinsh_bufs = tuple(PinshBuffer(shape, dtype) for shape, dtype in zip(shapes, dtypes))
        mpv_read_done = mp.Event()
        mpv_write_done = mp.Event()
        mpv_buf_dirty = mp.Event()
        row_views_lst: list[tuple[torch.Tensor, ...]] = []
        n_samples = pinsh_bufs[0].__len__()
        for i in range(n_samples):
            row_views_lst.append(tuple(pinsh_buf.pinsh_rows[i] for pinsh_buf in pinsh_bufs))  # noqa: PERF401
        row_views = tuple(row_views_lst)
        return cls(pinsh_bufs, mpv_read_done, mpv_write_done, mpv_buf_dirty, row_views)


@dataclass
class CuBufDcl:
    """Dataclass for multi-component CUDA tensor."""

    buffers: tuple[torch.Tensor, ...]
    mpv_read_queued: synchronize.Event
    mpv_write_queued: synchronize.Event
    mpv_buf_dirty: synchronize.Event
    cuv_read_done: _CudaEventBase
    cuv_write_done: _CudaEventBase
    custr_compute: _CudaStreamBase
    custr_transfer: _CudaStreamBase
    first_write: bool

    @classmethod
    def fromscratch(
        cls,
        shapes: Sequence[tuple[int, ...]],
        dtypes: Sequence[torch.dtype],
        compute_stream: _CudaStreamBase,
        transfer_stream: _CudaStreamBase,
    ) -> Self:
        """Create a MultiCuTensDcl object from scratch."""
        tensors = tuple(
            torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in zip(shapes, dtypes)
        )
        mpv_read_queued = mp.Event()
        mpv_write_queued = mp.Event()
        mpv_buf_dirty = mp.Event()
        cuv_read_done = torch.cuda.Event()
        cuv_write_done = torch.cuda.Event()
        return cls(
            tensors,
            mpv_read_queued,
            mpv_write_queued,
            mpv_buf_dirty,
            cuv_read_done,
            cuv_write_done,
            compute_stream,
            transfer_stream,
            first_write=True,
        )


"""Provide contexts that ease the pre and post actions for different operations and structures during H2D."""


class PinshReadCtx(contextlib.AbstractContextManager):
    """A context manager for reading from a PinnedSharedBuffer."""

    def __init__(self, dcl: PinshDcl) -> None:
        self.dcl = dcl

    def __enter__(self) -> Self:
        safewait(self.dcl.mpv_write_done)
        self.dcl.mpv_write_done.clear()
        return self

    def __exit__(self, *args) -> None:  # noqa: ANN002
        self.dcl.mpv_buf_dirty.clear()
        self.dcl.mpv_read_done.set()

    @property
    def buffers(self) -> tuple[PinshBuffer, ...]:
        """Return the buffer."""
        return self.dcl.buffers

    @property
    def is_dirty(self) -> bool:
        """Return the dirty flag."""
        return self.dcl.mpv_buf_dirty.is_set()


class PinshWriteCtx(contextlib.AbstractContextManager):
    """A context manager for writing to a PinnedSharedBuffer."""

    def __init__(self, dcl: PinshDcl) -> None:
        self.dcl = dcl

    def __enter__(self) -> Self:
        if not self.dcl.flag_resetted:
            safewait(self.dcl.mpv_read_done)
            self.dcl.mpv_read_done.clear()
        else:
            self.dcl.flag_resetted = False
        return self

    def __exit__(self, *args) -> None:  # noqa: ANN002
        self.dcl.mpv_write_done.set()

    @property
    def buffers(self) -> tuple[PinshBuffer, ...]:
        """Return the buffer."""
        return self.dcl.buffers

    def set_dirty(self) -> None:
        """Set the dirty flag."""
        self.dcl.mpv_buf_dirty.set()


class CuBufReadCtx(contextlib.AbstractContextManager):
    """A context manager for reading from a CUDA Tensor."""

    def __init__(self, dcl: CuBufDcl, *, host_sync_on_exit: bool = False) -> None:
        self.dcl = dcl
        self._host_sync_on_exit = host_sync_on_exit

    def pre_wait(self) -> None:
        """Wait for the mpv_write_queued event to be set."""
        safewait(self.dcl.mpv_write_queued)

    def __enter__(self) -> Self:
        safewait(self.dcl.mpv_write_queued)
        self.dcl.mpv_write_queued.clear()
        self.dcl.cuv_write_done.wait(self.dcl.custr_compute)  # safe wait for it is on GPU
        return self

    def __exit__(self, *args) -> None:  # noqa: ANN002
        self.dcl.cuv_read_done.record(self.dcl.custr_compute)
        self.dcl.mpv_read_queued.set()
        self.dcl.mpv_buf_dirty.clear()
        if self._host_sync_on_exit:
            self.dcl.cuv_write_done.synchronize()  # unsafe wait but little choice

    @property
    def buffers(self) -> tuple[torch.Tensor, ...]:
        """Return the tensor."""
        return self.dcl.buffers

    @property
    def is_dirty(self) -> bool:
        """Return the dirty flag."""
        return self.dcl.mpv_buf_dirty.is_set()


class CuBufWriteCtx(contextlib.AbstractContextManager):
    """A context manager for writing to a CUDA Tensor."""

    def __init__(self, dcl: CuBufDcl, *, host_sync_on_exit: bool = False) -> None:
        self.dcl = dcl
        self._host_sync_on_exit = host_sync_on_exit

    def __enter__(self) -> Self:
        if not self.dcl.first_write:
            safewait(self.dcl.mpv_read_queued)
            self.dcl.mpv_read_queued.clear()
            self.dcl.cuv_read_done.wait(self.dcl.custr_transfer)  # safe wait for it is on GPU

        else:
            self.dcl.first_write = False
        return self

    def __exit__(self, *args) -> None:  # noqa: ANN002
        self.dcl.cuv_write_done.record(self.dcl.custr_transfer)
        self.dcl.mpv_write_queued.set()
        if self._host_sync_on_exit:
            self.dcl.cuv_write_done.synchronize()  # unsafe wait but little choice

    @property
    def buffers(self) -> tuple[torch.Tensor, ...]:
        """Return the tensor."""
        return self.dcl.buffers

    def set_dirty(self) -> None:
        """Set the dirty flag."""
        self.dcl.mpv_buf_dirty.set()
