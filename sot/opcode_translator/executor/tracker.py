from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

from ...utils import InnerError, NameGenerator
from .guard import StringifyExpression, union_free_vars

if TYPE_CHECKING:
    from .pycode_generator import PyCodeGen
    from .variables import VariableBase


class Tracker:
    """
    Tracker is a base class responsible for tracking variables or objects in Python code.

    Each tracker is associated with a list of variables it tracks and a unique identifier.

    It is designed to generate instructions based on the variables, trace their values,
    and determine if they are traceable. Subclasses should implement their specific behaviors
    by overriding the `gen_instructions`, `trace_value_from_frame` and `is_traceable` methods.

    Args:
        inputs: The list of variables to be tracked.

    NOTE: It serves as an abstract class and should not be instantiated directly.
    """

    inputs: List[VariableBase]
    name_generator = NameGenerator("tracker_")

    def __init__(self, inputs: List[VariableBase]):
        """
        Initialize the Tracker.

        Args:
            inputs: The list of variables to be tracked.
        """
        self.inputs = inputs
        self.id = Tracker.name_generator.next()

    def gen_instructions(self, codegen: PyCodeGen) -> None:
        """
        Generate instructions based on the tracked variables. This is an abstract method
        and should be implemented by subclasses.

        Args:
            codegen (PyCodeGen): An instance of PyCodeGen to generate instructions.

        Raises:
            NotImplementedError: If this method is not overridden by a subclass.
        """
        raise NotImplementedError()

    def trace_value_from_frame(self) -> StringifyExpression:
        """
        Trace the value of the tracked variables from the frame. This is an abstract method
        and should be implemented by subclasses.

        Returns:
            The value of the tracked variables.

        Raises:
            NotImplementedError: If this method is not overridden by a subclass.
        """
        raise NotImplementedError()

    def is_traceable(self) -> bool:
        """
        Determine if the tracked variables are traceable. By default, all inputs should be traceable.
        This method can be overridden by subclasses to change the behaviour.

        Returns:
            True if all tracked variables are traceable, False otherwise.
        """
        for input in self.inputs:
            if not input.tracker.is_traceable():
                return False
        return True


class DummyTracker(Tracker):
    def __init__(self, inputs: list[VariableBase]):
        super().__init__(inputs)

    def gen_instructions(self, codegen: PyCodeGen):
        raise InnerError("DummyTracker has no instructions")

    def trace_value_from_frame(self):
        raise InnerError("DummyTracker can't trace value from frame")

    def is_traceable(self):
        return False

    def __repr__(self) -> str:
        return f"DummyTracker(num_inputs={len(self.inputs)})"


class DanglingTracker(Tracker):
    def __init__(self):
        super().__init__([])

    def gen_instructions(self, codegen: PyCodeGen):
        raise InnerError("DanglingTracker has no instructions")

    def trace_value_from_frame(self):
        raise InnerError("DanglingTracker can't trace value from frame")

    def is_traceable(self):
        return False

    def __repr__(self) -> str:
        return "DanglingTracker()"


class LocalTracker(Tracker):
    """
    LocalTracker is a subclass of Tracker that specifically tracks local variables in Python code.

    It generates instructions and traces the value of a local variable from the frame.

    Args:
        name (str): The name of the local variable to be tracked.
    """

    def __init__(self, name: str):
        super().__init__([])
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen) -> None:
        codegen.gen_load_fast(self.name)

    def trace_value_from_frame(self) -> StringifyExpression:
        return StringifyExpression(f"frame.f_locals['{self.name}']", {})

    def __repr__(self) -> str:
        return f"LocalTracker(name={self.name})"


class GlobalTracker(Tracker):
    """
    GlobalTracker is a subclass of Tracker that specifically tracks global variables in Python code.

    It generates instructions and traces the value of a global variable from the frame.

    Args:
        name (str): The name of the global variable to be tracked.
    """

    def __init__(self, name: str):
        super().__init__([])
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen) -> None:
        codegen.gen_load_global(self.name)

    def trace_value_from_frame(self) -> StringifyExpression:
        return StringifyExpression(f"frame.f_globals['{self.name}']", {})

    def __repr__(self) -> str:
        return f"GlobalTracker(name={self.name})"


class BuiltinTracker(Tracker):
    """
    BuiltinTracker is a subclass of Tracker that specifically tracks built-in variables in Python code.

    It generates instructions and traces the value of a built-in variable from the frame.

    Args:
        name (str): The name of the built-in variable to be tracked.
    """

    def __init__(self, name: str):
        super().__init__([])
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen) -> None:
        codegen.gen_load_global(self.name)

    def trace_value_from_frame(self) -> StringifyExpression:
        return StringifyExpression(
            f"builtins.__dict__[{self.name}]", {"builtins": builtins}
        )

    def __repr__(self) -> str:
        return f"BuiltinTracker(name={self.name})"


class ConstTracker(Tracker):
    def __init__(self, value):
        super().__init__([])
        self.value = value

    def gen_instructions(self, codegen: PyCodeGen):
        codegen.gen_load_const(self.value)

    def trace_value_from_frame(self):
        return StringifyExpression(f"{self.value}", {})

    def __repr__(self) -> str:
        return f"ConstTracker(value={self.value})"


class GetAttrTracker(Tracker):
    def __init__(self, obj: VariableBase, attr: str):
        super().__init__([obj])
        self.obj = obj
        self.attr = attr

    def gen_instructions(self, codegen: PyCodeGen):
        self.obj.tracker.gen_instructions(codegen)
        codegen.gen_load_attr(self.attr)

    def trace_value_from_frame(self):
        obj_tracer = self.obj.tracker.trace_value_from_frame()
        if self.attr.isidentifier():
            expr = f"{obj_tracer.expr}.{self.attr}"
        else:
            expr = f"getattr({obj_tracer.expr}, '{self.attr}')"
        return StringifyExpression(
            expr,
            union_free_vars(obj_tracer.free_vars),
        )

    def __repr__(self) -> str:
        return f"GetAttrTracker(attr={self.attr})"


class GetItemTracker(Tracker):
    def __init__(self, container_var: VariableBase, key: object):
        super().__init__([container_var])
        self.container = container_var
        self.key = key

    def gen_instructions(self, codegen: PyCodeGen):
        self.container.tracker.gen_instructions(codegen)
        codegen.gen_load_const(self.key)
        codegen.gen_subscribe()

    def trace_value_from_frame(self):
        container_tracer = self.container.tracker.trace_value_from_frame()
        return StringifyExpression(
            f"{container_tracer.expr}[{self.key!r}]",
            union_free_vars(container_tracer.free_vars),
        )

    def __repr__(self) -> str:
        return f"GetItemTracker(key={self.key!r})"


class GetIterTracker(Tracker):
    def __init__(self, iter_source: VariableBase):
        super().__init__([iter_source])
        self.iter_source = iter_source

    def gen_instructions(self, codegen: PyCodeGen):
        self.iter_source.tracker.gen_instructions(codegen)
        codegen._add_instr("GET_ITER")

    def trace_value_from_frame(self):
        iter_source_tracer = self.iter_source.tracker.trace_value_from_frame()
        return StringifyExpression(
            f"iter({self.iter_source.expr})",
            union_free_vars(iter_source_tracer.free_vars),
        )

    def __repr__(self) -> str:
        return "GetIterTracker()"
