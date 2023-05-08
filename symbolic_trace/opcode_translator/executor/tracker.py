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


class DummyTracker(Tracker):
    def __init__(self, inputs: list[VariableTracker]):
        super().__init__(inputs)

    def gen_instructions(self, codegen: PyCodeGen):
        raise InnerError("DummyTracker has no instructions")

    def trace_value_from_frame(self):
        raise InnerError("DummyTracker can't trace value from frame")


class LocalTracker(Tracker):
    def __init__(self, idx: int, name: str):
        super().__init__([])
        self.name = name
        self.idx = idx

    def gen_instructions(self, codegen: PyCodeGen):
        codegen._add_instr("LOAD_FAST", self.idx, self.name)

    def trace_value_from_frame(self):
        return lambda frame: frame.f_locals[self.name]


class GlobalTracker(Tracker):
    def __init__(self, name):
        super().__init__([])
        self.name = name


class ConstTracker(Tracker):
    def __init__(self, value):
        super().__init__([])
        self.value = value

    def gen_instructions(self, codegen: PyCodeGen):
        codegen.gen_load_const(self.value)

    def trace_value_from_frame(self):
        return lambda frame: self.value


class GetAttrTracker(Tracker):
    def __init__(self, obj_var: VariableTracker, attr_var: VariableTracker):
        super().__init__([])
        self.obj = obj_var
        self.attr = attr_var


class GetItemTracker(Tracker):
    def __init__(
        self, container_var: VariableTracker, key_var: VariableTracker
    ):
        super().__init__([key_var])
        self.container = container_var
        self.key = key_var

    def gen_instructions(self, codegen: PyCodeGen):
        from .variables import ConstantVariable

        self.container.tracker.gen_instructions(codegen)
        if isinstance(self.key.tracker, DummyTracker):
            assert isinstance(self.key, ConstantVariable)
            codegen.gen_load_const(self.key.value)
        else:
            self.key.tracker.gen_instructions(codegen)
        codegen._add_instr("BINARY_SUBSCR", 0, 0)

    def trace_value_from_frame(self):
        from .variables import ConstantVariable

        def trace_value(frame):
            container = self.container.tracker.trace_value_from_frame()(frame)
            if isinstance(self.key.tracker, DummyTracker):
                assert isinstance(self.key, ConstantVariable)
                key = self.key.value
            else:
                key = self.key.tracker.trace_value_from_frame()(frame)
            return container[key]

        return trace_value
