from __future__ import annotations

from typing import TYPE_CHECKING

from ...utils import InnerError

if TYPE_CHECKING:
    from .pycode_generator import PyCodeGen


def from_instruction(instr):
    pass


class Source:
    def __init__(self):
        pass

    def gen_instructions(self, codegen: PyCodeGen):
        raise NotImplementedError()

    def trace_value_from_frame(self):
        raise NotImplementedError()


class DummySource(Source):
    def __init__(self):
        pass

    def gen_instructions(self, codegen: PyCodeGen):
        raise InnerError("DummySource has no instructions")

    def gen_guard(self, value):
        raise InnerError("DummySource has no instructions")

    def trace_value_from_frame(self):
        raise InnerError("DummySource can't trace value from frame")


class LocalSource(Source):
    def __init__(self, idx: int, name: str):
        super().__init__()
        self.name = name
        self.idx = idx

    def gen_instructions(self, codegen: PyCodeGen):
        codegen._add_instr("LOAD_FAST", self.idx, self.name)

    def trace_value_from_frame(self):
        return lambda frame: frame.f_locals[self.name]


class GlobalSource(Source):
    def __init__(self, name):
        super().__init__()
        self.name = name


class ConstSource(Source):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def gen_instructions(self, codegen: PyCodeGen):
        codegen.gen_load_const(self.value)

    def trace_value_from_frame(self):
        return lambda frame: self.value


class GetAttrSource(Source):
    def __init__(self, obj_source, attr):
        super().__init__()
        self.attr = attr
        self.obj = obj_source


class GetItemSource(Source):
    def __init__(self, container_src: Source, key_src: Source):
        super().__init__()
        self.container_src = container_src
        self.key_src = key_src

    def gen_instructions(self, codegen: PyCodeGen):
        self.container_src.gen_instructions(codegen)
        self.key_src.gen_instructions(codegen)
        codegen._add_instr("BINARY_SUBSCR", 0, 0)

    def trace_value_from_frame(self):
        def trace_value(frame):
            container = self.container_src.trace_value_from_frame()(frame)
            key = self.key_src.trace_value_from_frame()(frame)
            return container[key]

        return trace_value
