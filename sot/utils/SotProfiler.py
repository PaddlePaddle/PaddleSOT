import atexit
import json
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps

_Profilers = set()


def _clear_profilers():
    profilers = set(_Profilers)
    for profiler in profilers:
        profiler.disable()


atexit.register(_clear_profilers)


class SotProfiler:
    def __init__(self, outpath=None):
        if outpath is None:
            self.outpath = sys.path[0]
        else:
            self.outpath = outpath
        self.event_roots = []
        self.event_stack = []

    def enable(self, tag=None):
        if self not in _Profilers:
            if tag is None:
                tag = "Main"
            _Profilers.add(self)
            self.event_roots.append(EventNode(EventMeta(tag)))
            self.event_stack = [self.event_roots[-1]]
            self.event_roots[-1].hold.start()

    def disable(self, dump=True):
        if self in _Profilers:
            self.event_roots[-1].hold.end()
            _Profilers.remove(self)

        if dump:
            self.dump_json()

    def push_event(self, event):
        node = self.event_stack[-1].push_event(event)
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

        json_infos = [
            build_json(root, root.end_time, {}) for root in self.event_roots
        ]

        with open(self.outpath + "/SotProfile.json", "w") as fp:
            json.dump(json_infos, fp, indent=4)

        print(
            f"[SotProfiler] JSON dumped to {self.outpath + '/SotProfile.json'}"
        )


@contextmanager
def ProfileGuard(outpath=None):
    profiler = SotProfiler(outpath)
    try:
        profiler.enable()
        yield
    finally:
        profiler.disable()


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

    def push_event(self, event):
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
    if not _Profilers or event_level < int(os.environ.get("EVENT_LEVEL", "0")):
        return None
    new_event = EventMeta(event_name)
    for profile in _Profilers:
        profile.push_event(new_event)
    new_event.start()
    return new_event


def event_end(event):
    if event is not None:
        event.end()
        for profile in _Profilers:
            profile.pop_event(event)


def event_register(event_name, event_level=0):
    def event_wrapper(func):
        @wraps(func)
        def call_with_event(*args, **kwargs):
            new_event = event_start(event_name, event_level)
            try:
                return func(*args, **kwargs)
            finally:
                event_end(new_event)

        return call_with_event

    return event_wrapper


@contextmanager
def EventGuard(event_name, event_level=0):
    try:
        new_event = event_start(event_name, event_level)
        yield
    finally:
        event_end(new_event)


def event_str(name, start_time, end_time, lasted):
    return f"[Event: {name}](start: {start_time}, end: {end_time}, lasted: {lasted})"
