"""To enable child processes to autoshutdown when the parent process dies.

A frequent problem in multiprocessing is that the parent process dies,
but the child process continues to run. If the child process holds resources
like file handles, sockets, memory, GPU memory, etc., then these resources
often cannot be reclaimed by the OS until the child process dies. Tracking down
the child process is sometimes difficult. In rare cases, a full system reboot
is required to reclaim the resources.

Just instantiate the DMS class in the child process, and it will
raise an exception if the parent process dies. This exception can be caught
and handled as needed. The DMS class uses a thread to check the
parent process ID every second. If the parent process ID changes, then the
exception is raised.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from multiprocessing import Process, current_process, parent_process
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import FrameType

MP_CURR_PROC = current_process()


class DMSError(RuntimeError):
    """Stub, to raise a custom exception when the DMS detects that the PPID has changed."""

    def __init__(
        self,
        msg: str = f"DMS: Process name {MP_CURR_PROC.name} / id {MP_CURR_PROC.pid} reparented.",
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        """To initialize the DMSError stub."""
        super().__init__(msg, *args, **kwargs)


class DMS:
    """To watch the parent process ID and raise an exception if it changes.

    Just instantiate this class in the child process, and it will raise an exception
    if the parent process dies. This exception can be caught and handled as needed.
    The DMS class uses a separate thread to check the parent process ID
    every second. If the parent process ID changes, then the exception is raised.

    The separate thread is used to avoid blocking the main process. That said, one must be careful
    to avoid deleting that thread, especially if you are managing other threads in the child process.
    """

    TIMEOUT = 1

    def __init__(self) -> None:
        """To initialize the DMS."""
        self.orig_ppid = os.getppid()
        self.pid = os.getpid()
        self.td_semaphore = threading.Semaphore(1)
        self.td_semaphore.acquire()
        self.td_thread = threading.Thread(target=self.dms_threadfn)
        self.td_thread.start()
        orig_parent_process = parent_process()
        self.orig_parent_name = orig_parent_process.name if orig_parent_process is not None else None
        signal.signal(signal.SIGUSR2, self.dms_exception_handler)

    def dms_threadfn(self) -> None:
        """To check the parent process ID every second. Runs in a separate thread."""
        while True:
            time.sleep(self.TIMEOUT)
            if os.getppid() != self.orig_ppid or self.td_semaphore.acquire(blocking=False):
                signal.raise_signal(signal.SIGUSR2)

    def manual_raise(self) -> None:
        """To manually raise the exception."""
        raise DMSError

    def dms_exception_handler(self, signum: int, frame: FrameType | None) -> None:  # noqa: ARG002
        """To raise the exception."""
        print("DMSERROR: Parent process ID changed. Raising exception.")
        raise DMSError


if __name__ == "__main__":
    """To test the DMS class."""

    def testing_dms_child() -> None:
        """To test the DMS class.

        This is a test of the child function. It creates the parent watcher and writes to a file.
        If you SIGKILL the parent, and the parent watcher is commented out, the child will continue
        to write to that file (you can see it). If the parent watcher is active, the child will
        raise an exception when the parent is killed. This exception will be printed to STDOUT unless
        it is caught and handled.
        """
        print("Child: name", current_process().name, "/ id", os.getpid())
        pa_wa = DMS()  # noqa: F841
        with open("test.txt", "w") as f:  # noqa: PTH123
            i = 0

            while True:
                # dms()
                r = 3.1415926535 * 10**2
                r = r * r
                r = r // 42
                f.write(f"{i}\n")
                f.flush()
                i += 1
                time.sleep(1)

    def testing_dms_main() -> None:
        """To test the DMS class.

        This is the main. After printing its PID, it creates the child process then sleeps.
        You can SIGKILL the parent and observe the behavior of the child process.
        """
        print(f"Main: name {current_process().name} / id {os.getpid()}")
        proc = Process(target=testing_dms_child)
        proc.start()
        print("Child started. Kill from outside.")
        time.sleep(1000)

    testing_dms_main()
