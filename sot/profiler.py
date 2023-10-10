import os
from contextlib import contextmanager
from functools import wraps

from paddle.framework import core

_event_level = int(os.environ.get("EVENT_LEVEL", "-1"))


class SotProfiler:
    def __enter__(self):
        self.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()

    def enable(self, tag=None):
        core.nvprof_start()
        core.nvprof_enable_record_event()

    def disable(self):
        core.nvprof_stop()


@contextmanager
def EventGuard(event_name, event_level=0):
    try:
        global _event_level
        need_pop = False
        if _event_level >= event_level:
            core.nvprof_nvtx_push(event_name)
            need_pop = True
        yield
    finally:
        if need_pop:
            core.nvprof_nvtx_pop()


if _event_level == -1:

    @contextmanager
    def _EmptyEventGuard(event_name, event_level=0):
        yield

    EventGuard = _EmptyEventGuard  # noqa: F811


def event_register(event_name, event_level=0):
    def event_wrapper(func):
        @wraps(func)
        def call_with_event(*args, **kwargs):
            with EventGuard(event_name, event_level=0):
                return func(*args, **kwargs)

        return call_with_event

    def do_nothing(func):
        return func

    global _event_level
    if _event_level >= event_level:
        return event_wrapper
    else:
        return do_nothing
