import atexit
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from functools import wraps

from paddle.framework import core

_enable_nvtx_record_event = False
_Profilers = set()
_event_level = int(os.environ.get("EVENT_LEVEL", "-1"))


def _clear_profilers():
    profilers = set(_Profilers)
    for profiler in profilers:
        profiler.disable(dump=True)


atexit.register(_clear_profilers)


class SotProfiler:
    def __init__(self, outpath=None):
        if outpath is None:
            self.outpath = (
                sys.path[0] + f'/SotProfile_{random.randint(0,999)}.json'
            )
        else:
            self.outpath = outpath
        self.event_root = EventNode(EventMeta("Main"))
        self.event_stack = []

    def __enter__(self):
        self.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()

    def __del__(self):
        self.disable(dump=True)

    def enable(self, tag=None):
        if self not in _Profilers:
            if tag is None:
                tag = f"Record_{len(self.event_root.sub_events)}"
            _Profilers.add(self)
            record_event = self.event_root.push_event_meta(EventMeta(tag))
            self.event_stack = [record_event]
            record_event.hold.start()

    def disable(self, dump=False):
        if self in _Profilers:
            self.event_root.sub_events[-1].hold.end()
            _Profilers.remove(self)

            if dump:
                self.dump_json()

    def push_event_meta(self, event):
        node = self.event_stack[-1].push_event_meta(event)
        self.event_stack.append(node)

    def pop_event(self, event):
        if event is self.event_stack[-1].hold:
            self.event_stack.pop()

    def dump_json(self):
        def build_json(node, default_end, infos):
            infos["name"] = node.name
            infos["start_time"] = node.start_time
            infos["end_time"] = (
                node.end_time if node.end_time is not None else default_end
            )
            infos["lasted"] = infos["end_time"] - infos["start_time"]
            infos["sub_events"] = []

            for sub_node in node.sub_events:
                infos["sub_events"].append(
                    build_json(sub_node, default_end, {})
                )

            return infos

        self.event_root.hold.start_time = self.event_root.sub_events[
            0
        ].start_time
        self.event_root.hold.end_time = self.event_root.sub_events[-1].end_time

        json_infos = build_json(self.event_root, self.event_root.end_time, {})
        with open(self.outpath, "w") as fp:
            json.dump(json_infos, fp, indent=4)

        print("=" * 50)
        print(f"[SotProfiler] JSON dumped to {self.outpath}")
        print("=" * 50)


# the key infomations for events, there is only one EventMeta will be created when one Event triggered
class EventMeta:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.lasted = None

    def start(self):
        self.start_time = time.perf_counter()

    def end(self):
        if self.end_time is None:
            self.end_time = time.perf_counter()
            self.lasted = self.end_time - self.start_time

    def __repr__(self):
        return event_str(self.name, self.start_time, self.end_time, self.lasted)


# EventNode is used for build tree struct of Events, every profiler holds EventNode as roots (if enable and disable called multi times)
class EventNode:
    def __init__(self, event: EventMeta):
        self.hold = event
        self.sub_events = []

    def push_event_meta(self, event):
        self.sub_events.append(EventNode(event))
        return self.sub_events[-1]

    def __repr__(self):
        return self.hold.__repr__()

    @property
    def name(self):
        return self.hold.name

    @property
    def start_time(self):
        return self.hold.start_time

    @property
    def end_time(self):
        return self.hold.end_time

    @property
    def lasted(self):
        return self.hold.lasted


def event_start(event_name, event_level=0):
    global _event_level
    if _Profilers and _event_level >= event_level:
        new_event = EventMeta(event_name)
        for profile in _Profilers:
            profile.push_event_meta(new_event)
        new_event.start()
        return new_event
    return None


def event_end(event):
    if event is not None:
        event.end()
        for profile in _Profilers:
            profile.pop_event(event)


@contextmanager
def EventGuard(event_name, event_level=0):
    try:
        new_event = event_start(event_name, event_level)
        yield
    finally:
        event_end(new_event)


class _NvtxProfiler:
    def __enter__(self):
        self.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()

    def enable(self, tag=None):
        global _enable_nvtx_record_event
        assert (
            _enable_nvtx_record_event is False
        ), "Nvtx event already enabled, do not use multi NvtxProfilers!"
        core.nvprof_start()
        core.nvprof_enable_record_event()
        _enable_nvtx_record_event = True

    def disable(self):
        global _enable_nvtx_record_event
        assert (
            _enable_nvtx_record_event is True
        ), "Can not stop NvtxProfiler when it was not started!"
        core.nvprof_stop()
        _enable_nvtx_record_event = False


@contextmanager
def _NvtxEventGuard(event_name, event_level=0):
    try:
        global _enable_nvtx_record_event, _event_level
        need_pop = False
        if _enable_nvtx_record_event and _event_level >= event_level:
            core.nvprof_nvtx_push(event_name)
            need_pop = True
        yield
    finally:
        if need_pop:
            core.nvprof_nvtx_pop()


if os.environ.get("USE_JSON_PROFILE", "False") != "True":
    EventGuard = _NvtxEventGuard  # noqa: F811
    SotProfiler = _NvtxProfiler  # noqa: F811


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


def event_str(name, start_time, end_time, lasted):
    return f"[Event: {name}](start: {start_time}, end: {end_time}, lasted: {lasted})"


@contextmanager
def sotprof_range(iter_id, start, end, exit_after_prof=True):
    """
    this interface is almost same to paddle.profile.utils._nvprof_range
    except this api will set _enable_nvtx_record_event=True (to enable sot events)
    """
    if start >= end:
        yield
        return

    try:
        if iter_id == start:
            _NvtxProfiler().enable()
            core.nvprof_enable_record_event()
        if iter_id >= start:
            core.nvprof_nvtx_push(str(iter_id))
        yield
    finally:
        if iter_id < end:
            core.nvprof_nvtx_pop()
        if iter_id == end - 1:
            _NvtxProfiler().disable()
            if exit_after_prof:
                sys.exit()
