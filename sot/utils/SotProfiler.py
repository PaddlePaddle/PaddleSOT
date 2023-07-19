import atexit
import os
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

    def disable(self):
        if self in _Profilers:
            self.event_roots[-1].hold.end()
            _Profilers.remove(self)

    def push_event(self, event):
        node = self.event_stack[-1].push_event(event)
        self.event_stack.append(node)

    def pop_event(self, event):
        if event is self.event_stack[-1].hold:
            self.event_stack.pop()

    def __del__(self):
        self.report()

    def report(self):
        def collect_stat(node, default_end, stat_info, tree_info, prefix):
            if node.name not in stat_info:
                stat_info[node.name] = 0

            if node.lasted is None:
                lasted = default_end - node.start_time
                tree_info.append(
                    prefix
                    + event_str(node.name, node.start_time, default_end, lasted)
                )
                stat_info[node.name] += lasted
            else:
                tree_info.append(
                    prefix
                    + event_str(
                        node.name, node.start_time, node.end_time, node.lasted
                    )
                )
                stat_info[node.name] += node.lasted

            for sub_node in node.sub_events:
                collect_stat(
                    sub_node, default_end, stat_info, tree_info, prefix + "    "
                )

        for root in self.event_roots:
            stat_info = {}
            tree_info = []
            collect_stat(root, root.hold.end_time, stat_info, tree_info, "")

            print("[Call Struct]:")
            print("\n".join(tree_info))

            print("\n[Stat Infos]:")
            for k, v in sorted(
                stat_info.items(), key=lambda kv: kv[1], reverse=True
            ):
                print(f"    {k:<20s}: {v}")


@contextmanager
def ProfileGuard(name=None, outpath=None):
    profiler = SotProfiler(name, outpath)
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
