"""To create a buffer that is both pinned and shared.

MOTIVATION

Pinned (page-locked) memory is that which cannot be paged out to disk.
It is guaranteed to be resident in the physical memory.

Pinned memory is significantly faster for host-to-device (H2D, CPU->GPU) transfers.
Chiefly, host non-blocking and device non-blocking transfers cannot occur without
the host memory being pinned beforehand.

In the Pytorch dataloader as of v2.4, the dataloader workers (child processes) write to paged memory.
Then, a background thread in the main process reads from that paged memory and copies it to pinned memory.
This is a two-step process, and one of those steps involves the main process.

Note also that we are still in Pytorch 2.4 / Python 3.12 , and the no-GIL implementation of Python
is still very experimental. Many features of Pytorch (such as torch.compile) are not yet supported
in this mode.

Further, memory copies from host RAM to host RAM are not zero-cost for the host.
There is a significant processing overhead for this, which is surprising given that H2D-async
and storage-to-host copies cost very little.

So even if there are sufficient workers, there is a crucial time bottleneck within the main process.

1. Paged to pinned copy.
2. H2D copy that depends on 1) and is blocking if it is not scheduled well in advance.

Note also that the CPU is not idle apart from dataloading.
Compiling CUDA kernels, Python calls' overhead, etc affect the CPU.

IMPLEMENTATION

CUDA runtime contexts cannot be shared across processes. However, CUDA IPC exists.

The naive solution is to launch an instance of the CUDA Runtime in every worker.
This would quickly cause the host to run out of RAM, because each instance of the
CUDA runtime takes up 2GB of memory. With, say, 8 workers, you're already 16GB in the red.

Broadly, CUDA (as of Runtime API 6.11) provides two mechanisms for registering host memory as pinned:

1. cudaHostAlloc -- the CUDA runtime takes unused memory, and returns this new, empty block to the program.
It can be thought of as a C-style malloc, but for CUDA host pinned memory.

2. cudaHostRegister -- the program provides a pointer and a size, and the runtime pins that block of
memory. The key thing here is that the program i.e. we must provide the memory to CUDA. And then it pins it.

What if we pass shared memory to this? The other processes don't care about CUDA,so we don't incur
that memory penalty.

It works. At least, it works on Linux.
There are caveats, particularly around size (varying from OS to OS).
Hence the design decision to _not_ subclass torch.Tensor right away.

'Pinsh' is short for 'pinned and shared', throughout this module.

Since the max size of contiguous pinned mem is a limited at OS level,
we need to split the tensor into smaller tensors.

This is why the word "buffer" is used instead of "tensor",
and why the class is not a subclass of torch.Tensor.

Note that none of the methods of torch.Tensor apply here.
"""
# FROZEN

from __future__ import annotations

import atexit
from math import prod
from typing import TYPE_CHECKING, Self

import cuda.bindings.runtime as nv_cudart  # type: ignore
import torch

_SUPPRESS_CUDARETVALERROR = False


def disable_cudart_error() -> None:
    """Suppress the CudartRetvalError exception."""
    global _SUPPRESS_CUDARETVALERROR  # noqa: PLW0603
    _SUPPRESS_CUDARETVALERROR = True


def enable_cudart_error() -> None:
    """Enable the CudartRetvalError exception."""
    global _SUPPRESS_CUDARETVALERROR  # noqa: PLW0603
    _SUPPRESS_CUDARETVALERROR = False


if TYPE_CHECKING:
    from collections.abc import Sequence


_MAX_PINSHM_TENS_BYTES = 1 << 30
_PERTENS_PINSIZE_MSG = "Pinned shared memory {} is larger than the maximum allowed size of {}.\
    This is likely to cause a CUDA error. You can try increasing _MAX_PERTENS_PINSIZE,\
    but this does not guarantee success. The final limit is enforced by CUDA and the OS."


class CudartRetvalError(RuntimeError):
    """Exception raised for nv_cudart.cudaError_t return values with nonzero return values."""


class PinshBuffer:
    """Create a shared, page-locked (pinned) buffer of tensor chunks."""

    def __init__(self, shape: Sequence[int], dtype: torch.dtype) -> None:
        """Initialize a new, empty PinshBuffer.

        Args:
            shape (Sequence[int]): Shape of the tensor.
            dtype (torch.dtype): Data type of the tensor.

        """
        ########1. Set the constants.
        self._shape = shape
        self._dtype = dtype
        self._n_bytes = prod(self._shape) * self._dtype.itemsize
        self._len = self._shape[0]
        self.sample_shape = self._shape[1:]

        ########2. Create paged tensors.
        basic_tens_lst = []

        if self._n_bytes < _MAX_PINSHM_TENS_BYTES:  # If desired mem is small, don't split.
            basic_tens_lst.append(torch.zeros(*self._shape, dtype=self._dtype))

        else:  # Else make tensors not bigger than _MAX_PINSHM_TENS_BYTES
            n_rows = self._len
            n_bytes_per_row = self._dtype.itemsize * prod(self.sample_shape)
            n_rows_per_chunk = _MAX_PINSHM_TENS_BYTES // n_bytes_per_row
            i = 0
            while i < n_rows:
                dx = min(n_rows - i, n_rows_per_chunk)
                basic_tens_lst.append(torch.zeros(dx, *self.sample_shape, dtype=self._dtype))
                i += dx

        ########3. Convert the paged tensors from 2. to pinsh tensors.
        pinsh_chunks: list[torch.Tensor] = []
        self._ptrs: list[int] = []
        for btens in basic_tens_lst:
            pinsh_tens, stor_ptr = self._convert_to_pinsh(btens)
            pinsh_chunks.append(pinsh_tens)
            self._ptrs.append(stor_ptr)
        self._pinsh_chunks = tuple(pinsh_chunks)
        ########4. Make a list of row views of the pinsh tensors.
        # Each row is one sample in the dataset/batch.
        rowviews_lst: list[torch.Tensor] = []
        for pinsh_tens in self._pinsh_chunks:
            rowviews_lst.extend([pinsh_tens[i] for i in range(pinsh_tens.shape[0])])
        self._pinsh_rows = tuple(rowviews_lst)
        atexit.register(self.__del__)

    def copy_(self, tens: torch.Tensor, *, non_blocking: bool = False) -> None:
        """Copy to the PinShBuffer from the given tensor."""
        x = 0
        dx = 0
        for i in range(len(self._pinsh_chunks)):
            split = self._pinsh_chunks[i]
            dx = split.shape[0]
            split.copy_(tens[x : x + dx], non_blocking=non_blocking)
            x += dx

    def copyto(self, tens: torch.Tensor, *, non_blocking: bool = False) -> None:
        """Copy the PinShBuffer to the given tensor."""
        x = 0
        dx = 0
        for i in range(len(self._pinsh_chunks)):
            split = self._pinsh_chunks[i]
            dx = split.shape[0]
            tens[x : x + dx].copy_(split, non_blocking=non_blocking)
            x += dx

    def __len__(self) -> int:
        """Get the length of the tensor."""
        return self._len

    def __del__(self) -> None:
        """Call cudaHostUnregister on the given pinsh tensor pointers -- but only if CUDA is initialized."""
        if torch.cuda.is_initialized():
            atexit.unregister(self.__del__)
            self._cudart_unregister_ptrs(self._ptrs)
        else:
            print("Skipping cudaHostUnregister because CUDA is not initialized.")

    @property
    def pinsh_rows(self) -> tuple[torch.Tensor, ...]:
        """To get the rows of the buffer."""
        return self._pinsh_rows

    @property
    def pinsh_chunks(self) -> tuple[torch.Tensor, ...]:
        """To get the chunks of the buffer."""
        return self._pinsh_chunks

    @classmethod
    def _convert_to_pinsh(cls: type[Self], tens: torch.Tensor) -> tuple[torch.Tensor, int]:
        """To convert a given tensor which is both shared and pinned.

        First, Pytorch's mechanisms are used to create shared memory.
        Then, that underlying storage is pinned through the CUDA Runtime API.
        (torch 's default allocator doesn't allow simultaneously pinned and shared memory, somehow.)

        Note that there is an upper bound on the size of each pinned_shared block and also
        on the cumulative size of all pinned_shared blocks.
        """
        if not tens.is_cpu:
            msg = "Tensor must be on CPU to convert to shared memory."
            raise ValueError(msg)
        if not tens.is_contiguous():
            msg = "Tensor must be contiguous to convert to shared memory."
            raise ValueError(msg)

        stor = tens.untyped_storage()
        stor.share_memory_()
        stor_size, stor_ptr = stor.size(), stor.data_ptr()

        if stor_size > _MAX_PINSHM_TENS_BYTES:
            msg = _PERTENS_PINSIZE_MSG.format(stor_size, _MAX_PINSHM_TENS_BYTES)
            raise ValueError(msg)
        cls._handle_cuda_error(nv_cudart.cudaHostRegister(stor_ptr, stor_size, 0))
        cls._check_is_pinsh_tensor(tens)
        return tens, stor_ptr

    @classmethod
    def _handle_cuda_error(cls, cu_err: nv_cudart.cudaError_t | tuple[nv_cudart.cudaError_t, ...]) -> None:
        """To check for errors in nv_cudart function calls."""
        if not _SUPPRESS_CUDARETVALERROR:
            err: nv_cudart.cudaError_t = cu_err[0] if isinstance(cu_err, tuple) else cu_err

            if not isinstance(err, nv_cudart.cudaError_t):
                msg = "Last nv_cudart call returned something other than a nv_Cudart.cudaError_t .\
                    This is highly unusual."
                raise RuntimeError(msg)  # this IS a result of some major runtime issue, not just type.
            if err.value != 0:
                msg = f"Last nv_cudart call returned error code {err.value} with name {err.name}."
                raise CudartRetvalError(msg)

    @classmethod
    def _check_is_pinsh_tensor(cls, tens: torch.Tensor) -> None:
        """Check if the given tensor is a shared and pinned tensor."""
        if not tens.is_shared() or not tens.is_pinned():
            msg = f"is_shared:{tens.is_shared()} / is_pinned:{tens.is_pinned()}"
            raise ValueError(msg)

    @classmethod
    def _cudart_unregister_ptrs(cls, lst: Sequence[int]) -> None:
        """To call cudaHostUnregister on the given pinsh tensor pointers."""
        for ptr in lst:
            cls._handle_cuda_error(nv_cudart.cudaHostUnregister(ptr))
