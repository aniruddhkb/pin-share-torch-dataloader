"""Provides a safer event waiting mechanism -- threads that are waiting can handle exceptions and signals.

If a thread is waiting on an event in the usual way, it cannot receive signals or handle exceptions.
(SIGKILL will of course still work, but that's not the point).

However, polling with a timeout on the event would enable the "waiter" to react to other things as well.
"""

from __future__ import annotations

import threading
from multiprocessing import synchronize

DEFAULT_EVENT_WAIT = 1


def safewait(eve: synchronize.Event | threading.Event) -> None:
    """Wait for the event to be set."""
    if not isinstance(eve, (synchronize.Event, threading.Event)):
        msg = "eve must be a threading.Event or multiprocessing.Event"
        raise TypeError(msg)
    while not eve.wait(DEFAULT_EVENT_WAIT):
        pass
