from __future__ import annotations

from typing import TYPE_CHECKING

from ...utils import InnerError

if TYPE_CHECKING:
    from .pycode_generator import PyCodeGen
    from .variables import VariableTracker


def from_instruction(instr):
    pass


class Tracker:
    inputs: list[VariableTracker]

    def __init__(self, inputs: list[VariableTracker]):
        self.inputs = inputs

    def gen_instructions(self, codegen: PyCodeGen):
        raise NotImplementedError()

    def trace_value_from_frame(self):
        raise NotImplementedError()

    def is_traceable(self):
        for input in self.inputs:
            if not input.tracker.is_traceable():
                return False
        return True


class DummyTracker(Tracker):
    def __init__(self, inputs: list[VariableTracker]):
        super().__init__(inputs)

    def gen_instructions(self, codegen: PyCodeGen):
        raise InnerError("DummyTracker has no instructions")

    def trace_value_from_frame(self):
        raise InnerError("DummyTracker can't trace value from frame")


class LocalTracker(Tracker):
    def __init__(self, name: str):
        super().__init__([])
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen):
        codegen.gen_load_fast(self.name)

    def trace_value_from_frame(self):
        return lambda frame: frame.f_locals[self.name]


class GlobalTracker(Tracker):
    def __init__(self, name):
        super().__init__([])
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen):
        codegen.gen_load_global(self.name)

    def trace_value_from_frame(self):
        def inner(frame):
            if self.name in frame.f_gloabls.keys():
                return frame.f_gloabls[self.name]
            else:
                return frame.f_builtins[self.name]

        return inner


class ConstTracker(Tracker):
    def __init__(self, value):
        super().__init__([])
        self.value = value

    def gen_instructions(self, codegen: PyCodeGen):
        codegen.gen_load_const(self.value)

    def trace_value_from_frame(self):
        return lambda frame: self.value


class GetAttrTracker(Tracker):
    def __init__(self, obj: VariableTracker, attr: str):
        super().__init__([])
        self.obj = obj
        self.attr = attr

    def gen_instructions(self, codegen: PyCodeGen):
        self.obj.tracker.gen_instructions(codegen)
        codegen.gen_load_attr(self.attr)


class GetItemTracker(Tracker):
    def __init__(self, container_var: VariableTracker, key: object):
        super().__init__([container_var])
        self.container = container_var
        self.key = key

    def gen_instructions(self, codegen: PyCodeGen):
        self.container.tracker.gen_instructions(codegen)
        codegen.gen_load_const(self.key)
        codegen.gen_subscribe()

    def trace_value_from_frame(self):
        def trace_value(frame):
            container = self.container.tracker.trace_value_from_frame()(frame)
            return container[self.key]

        return trace_value
