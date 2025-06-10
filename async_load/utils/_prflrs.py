"""Profilers with a common interface for rapid swapping.

This is to enable rapid swapping between:
- torch.profiler.profile
- nsys (Nsight Systems)
- no profiling (bypass)

Without requiring changes to the code being profiled (except for a flag).

MOTIVATION:

I frequently use both torch.profiler and nsys for the same loop, to analyze from
different perspectives. nsys cannot get the CPU part well.
unless it has very invasive privileges (which no cloud provider will give).

torch.profiler is far inferior to nsys on the CUDA API + GPU-HW side,
but tracks the python side well.

USAGE:

See "proftest.py" for a usage example. This mimics the API of torch.profiler.profile.
However, it is NOT a drop-in replacement, primarily to avoid name collisions.

INTERNALS:

In torch.profiler.profile, there are two nested contexts.

There is first the outer context which encapsulates the whole loop.
In Pytorch, this is the torch.profiler.profile instance.
Here, it is the class hierarchy of AbstractProfiler.

The second is the record_function context,
which provides annotations to the trace for the enclosed block.
In Pytorch, this is torch.profiler.record_function .
Here, it is the class hierarchy of _AbstractRecordFunctionContext.

In both cases, the interface needs to be followed verbatim.

AbstractProfiler:
- __init__ : Initialize the profiler.
- nextstep : Call this method at the end of the part to be profiled.
- record_function : The context manager to record a function.

_AbstractRecordFunctionContext:
- __init__ : Initialize the context manager.
- __enter__ : Enter the context manager.
- __exit__ : Exit the context manager.

"""

from __future__ import annotations

import os
from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Callable, Self, TypedDict

import torch
from torch.cuda import nvtx
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)


class TorchProfilerCompleteError(ValueError):
    """Exception to indicate that the profiler has completed."""


class _AbstractRecordFunctionContext(AbstractContextManager):
    @abstractmethod
    def __init__(self, name: str) -> None:
        pass


class _DummyRecordFunctionContext(_AbstractRecordFunctionContext):
    """To run the script without profiling."""

    def __init__(self, name: str) -> None:
        """Instantiate a DummyRecordFunctionContext object."""
        self.name = name

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        """Exit the context manager."""


class _PushNVTXRange(_AbstractRecordFunctionContext):
    """To add an NVTX range around a sequence of operations, in a new context."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> Self:
        nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        nvtx.range_pop()


class AbstractProfiler(AbstractContextManager):
    """Base class for profilers.

    All profiler classses must follow this interface verbatim.
    Otherwise the whole point of this is lost.
    """

    @abstractmethod
    def __init__(
        self,
        wait_steps: int = 0,
        warmup_steps: int = 0,
        capture_steps: int = 0,
        log_dir: str = "",
    ) -> None:
        """Instantiate a profiler object.

        Every subclass must have a self.record_function attribute.
        """

    @property
    @abstractmethod
    def nextstep(self) -> Any:  # noqa: ANN401
        """Call this method at the beginning of the part to be profiled."""

    @property
    @abstractmethod
    def record_function(self) -> Any:  # noqa: ANN401
        """The context manager to record a function."""


class DummyProfiler(AbstractProfiler):
    """To run the script without profiling.

    This is for hyperparameter tuning and long-run training/inference.
    """

    def __init__(
        self,
        wait_steps: int = 0,
        warmup_steps: int = 0,
        capture_steps: int = 0,
        log_dir: str = "",
    ) -> None:
        """Instantiate a DummyProfiler object."""

    @property
    def record_function(self) -> type[_AbstractRecordFunctionContext]:
        return _DummyRecordFunctionContext

    def _step(self) -> None:
        """Call this method at the beginning of the part to be profiled."""

    @property
    def nextstep(self) -> Callable[[], None]:
        """Call this method at the beginning of the part to be profiled."""
        return self._step

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        """Exit the context manager."""


class NsightSystemsProfiler(AbstractProfiler):
    """To profile with the CLI tool Nsight Systems.

    Run the python script run command through nsys. A script is provided in nspy_*.sh .
    You will need nsys installed on your system. Apart from the whole CUDA toolkit, of course.
    This class just starts and stops the profiler, and adds nvtx ranges.
    """

    def __init__(
        self,
        wait_steps: int,
        warmup_steps: int,
        capture_steps: int,
        log_dir: str = "",  # noqa: ARG002
    ) -> None:
        """Instantiate a NsightSystemsProfiler object."""
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.capture_steps = capture_steps
        self.currstep = 0
        self.started = False
        self.stopped = False

    @property
    def record_function(self) -> type[_PushNVTXRange]:
        return _PushNVTXRange

    @property
    def nextstep(self) -> Callable[[], None]:
        return self._step

    def _step(self) -> None:
        """Call this method at the beginning of the part to be profiled."""
        self.currstep += 1
        if not self.started and self.currstep >= self.wait_steps + self.warmup_steps:
            print("akb: Starting Nsight Systems profiler.")
            torch.cuda.cudart().cudaProfilerStart()  # type: ignore
            self.started = True

        if (
            self.started
            and not self.stopped
            and self.currstep > self.wait_steps + self.warmup_steps + self.capture_steps
        ):
            print("akb: Stopping Nsight Systems profiler.")
            torch.cuda.cudart().cudaProfilerStop()  # type: ignore
            self.stopped = True

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        """Exit the context manager."""


class TorchProfiler(AbstractProfiler, profile):
    """To profile with torch.profiler."""

    def __init__(
        self,
        wait_steps: int = 0,
        warmup_steps: int = 0,
        capture_steps: int = 0,
        log_dir: str = "",
        *,
        no_cuda: bool = False,
    ) -> None:
        """Instantiate a TorchProfiler object."""
        self._schedule = schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=capture_steps,
            repeat=1,
        )
        self._on_trace_ready = tensorboard_trace_handler(
            log_dir,
            use_gzip=True,
        )
        profile.__init__(  # frz
            self,
            schedule=self._schedule,
            activities=[ProfilerActivity.CPU]
            if no_cuda
            else [
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            on_trace_ready=self._on_trace_ready,
            record_shapes=False,
            profile_memory=False,
            with_stack=True,  # <--- this is the one you need. This and CPU and CUDA. but make it SHORT.
            with_flops=False,
            with_modules=False,  # appoears to cost nothing, but give nothing
        )
        self._record_function = record_function
        self._started = False
        self._currstep = 0
        self._max_total_steps = wait_steps + warmup_steps + capture_steps

    @property
    def record_function(self) -> Any:  # noqa: ANN401
        """The context manager to record a function."""
        return self._record_function

    def __enter__(self) -> Self:
        """Enter the context manager."""
        print("akb: Starting torch profiler.")
        to_return = profile.__enter__(self)
        self._started = True
        return to_return  # type: ignore

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        """Exit the context manager."""
        if self._started:
            self._started = False
            print("akb: Stopping torch profiler.")
            return profile.__exit__(self, exc_type, exc_value, traceback)
        # print("akb: Torch profiler was not started.")
        # This is to avoid the error when exiting the context manager
        # without starting the profiler.
        # This is a bit of a hack, but it works.
        # It is better than raising an exception.
        return None

    def __del__(self) -> None:
        """Exit the context manager."""
        if self._started:
            self.__exit__(None, None, None)

    @property
    def nextstep(self) -> Callable[[], None]:
        """Call this method at the beginning of the part to be profiled."""
        if self._currstep >= self._max_total_steps:
            self.__exit__(None, None, None)
        self._currstep += 1
        return self.step


def def_log_dir() -> str:
    """Get default log directory for profiling."""
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")  # noqa: DTZ005
    return os.path.join(
        "./traces",
        current_time + "_" + socket.gethostname(),
    )


class ProfilerArgs(TypedDict):
    """Profiler arguments."""

    mode: str
    wait_steps: int
    warmup_steps: int
    capture_steps: int
    log_dir: str
    bypassed: bool


def get_profiler(  # noqa: PLR0913
    mode: str,
    wait_steps: int,
    warmup_steps: int,
    capture_steps: int,
    log_dir: str,
    *,
    bypassed: bool,
) -> DummyProfiler | TorchProfiler | NsightSystemsProfiler:
    """Get the appropriate profiler object."""
    if bypassed:
        return DummyProfiler(wait_steps, warmup_steps, capture_steps, log_dir)
    if mode == "torch":
        return TorchProfiler(wait_steps, warmup_steps, capture_steps, log_dir)
    if mode == "torch_no_cuda":
        return TorchProfiler(wait_steps, warmup_steps, capture_steps, log_dir, no_cuda=True)
    if mode == "nsys":
        return NsightSystemsProfiler(wait_steps, warmup_steps, capture_steps, log_dir)

    erm = f"Invalid profiler mode {mode}."
    raise ValueError(erm)
