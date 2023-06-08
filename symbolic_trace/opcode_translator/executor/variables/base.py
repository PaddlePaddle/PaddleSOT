from __future__ import annotations

import inspect
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable

import paddle

from ....utils import NameGenerator, log, log_do
from ....utils.exceptions import InnerError, NotImplementException
from ..guard import StringifyExpression, union_free_vars
from ..pycode_generator import PyCodeGen
from ..tracker import DummyTracker, GetAttrTracker, Tracker

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph


ConstTypes = (int, float, str, bool, type(None))


def get_zero_degree_vars(
    variables: set[VariableBase], visited_vars: list[VariableBase]
) -> list[VariableBase]:
    return [
        var
        for var in variables
        if var not in visited_vars
        and len(set(var.get_traceable_inputs()) - set(visited_vars)) == 0
    ]


def topo_sort_vars(
    root_vars: list[VariableBase],
) -> list[VariableBase]:
    unique_vars = set()

    for var in root_vars:
        unique_vars |= set(var.flatten_traceable_inputs())

    topo_ordered_vars = []
    topo_queue = Queue()
    for var in get_zero_degree_vars(unique_vars, topo_ordered_vars):
        topo_queue.put(var)

    while not topo_queue.empty():
        var = topo_queue.get()
        topo_ordered_vars.append(var)
        for zero_degree_var in get_zero_degree_vars(
            unique_vars, topo_ordered_vars
        ):
            if (
                zero_degree_var in topo_queue.queue
                or zero_degree_var in topo_ordered_vars
            ):
                continue
            topo_queue.put(zero_degree_var)
    return topo_ordered_vars


def map_variables(map_func, variables):
    def _map_variable(variable):
        assert isinstance(
            variable, VariableBase
        ), f"variable must be VariableBase, got {variable}"
        from .container import ContainerVariable

        if isinstance(variable, ContainerVariable):
            return paddle.utils.map_structure(
                _map_variable, variable.get_wrapped_items()
            )
        return map_func(variable)

    return paddle.utils.map_structure(_map_variable, variables)


class VariableFactory:
    registered_funcs: dict[str, list[str]] = {"default": []}
    mapping_str_func: dict[str, Callable] = {}

    @staticmethod
    def default_from_value(value, graph, tracker):
        from .basic import ObjectVariable

        return ObjectVariable(value, graph, tracker)

    @staticmethod
    def register_from_value(*, successor: str | None = None):
        registered_funcs = VariableFactory.registered_funcs
        mapping_str_func = VariableFactory.mapping_str_func

        def _register_from_value(func: Callable):
            name = func.__qualname__.split(".")[0]
            mapping_str_func[name] = func
            if successor is None:
                registered_funcs["default"].append(name)
            elif successor not in registered_funcs.keys():
                registered_funcs[successor] = [name]
            else:
                registered_funcs[successor].append(name)

        log(4, VariableFactory.registered_funcs)
        return _register_from_value

    @staticmethod
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        registered_funcs = VariableFactory.registered_funcs

        def _find_var(key: str = "default"):
            for name in registered_funcs[key]:
                if name in registered_funcs.keys():
                    var = _find_var(name)
                    if var is not None:
                        return var
                func = VariableFactory.mapping_str_func[name]
                var = func(value, graph, tracker)
                if var is not None:
                    return var

        var = _find_var()
        if var is not None:
            return var
        return VariableFactory.default_from_value(value, graph, tracker)


class VariableBase:
    """
    VariableBase is a basic concept and each symbols in VM stack is regarded as
    an Variable Object in symblic tracing process.
    """

    tracker: Tracker
    name_generator = NameGenerator("object_")

    def __init__(self, tracker: Tracker):
        self.tracker = tracker
        self.id = VariableBase.name_generator.next()

    def __hash__(self):
        return hash(self.id)

    def make_stringify_guard(self) -> StringifyExpression:
        assert (
            self.tracker.is_traceable()
        ), "Cannot make guard from a non-traceable variable."

        frame_value_tracer = self.tracker.trace_value_from_frame()
        log_do(
            4,
            lambda: print(
                f"[Guard]: guard_fn for {self}, tracker={self.tracker.__class__.__name__}, value={frame_value_tracer.expr}"
            ),
        )
        return StringifyExpression(
            f"{frame_value_tracer.expr} == {self.get_value()}",
            union_free_vars(frame_value_tracer.free_vars),
        )

    def get_value(self) -> Any:
        raise NotImplementedError()

    def reconstruct(self, codegen: PyCodeGen):
        """
        Contruct an opcode and append it into codegen.instructions.
        """
        if (
            not isinstance(self.tracker, DummyTracker)
            and self.tracker.is_traceable()
        ):
            self.tracker.gen_instructions(codegen)
        else:
            self._reconstruct(codegen)

    def _reconstruct(self, codegen: PyCodeGen):
        raise NotImplementException()

    def flatten_items(self) -> list[VariableBase]:
        from .container import ContainerVariable

        if not isinstance(self, ContainerVariable):
            return [self]
        flattened_items = []
        for item in self.get_items():
            flattened_items.extend(item.flatten_items())
        return flattened_items

    def get_inputs(self) -> list[VariableBase]:
        return self.tracker.inputs

    def get_traceable_inputs(self) -> list[VariableBase]:
        if self.tracker.is_traceable():
            return []

        return list(
            filter(lambda x: x.tracker.is_traceable(), self.tracker.inputs)
        )

    def flatten_traceable_inputs(self) -> list[VariableBase]:
        if self.tracker.is_traceable():
            return [self]

        flattened_traceable_inputs: list[VariableBase] = []
        for input in self.get_inputs():
            flattened_traceable_inputs.extend(input.flatten_traceable_inputs())
        return flattened_traceable_inputs

    def call_function(self, *args, **kwargs):
        pass

    def __getattr__(self, name: str):
        if not hasattr(self.value, name):
            raise InnerError(
                f"{self.__class__.__name__} {self} has no attribute {name}"
            )
        attr = getattr(self.value, name)
        if inspect.ismethod(attr):
            from .callable import UserDefinedMethodVariable

            return UserDefinedMethodVariable(
                self,
                attr.__func__,
                graph=self.graph,
                tracker=GetAttrTracker(self, name),
            )
        return VariableFactory.from_value(
            attr, self.graph, tracker=GetAttrTracker(self, name)
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id})"

    def __str__(self):
        return self.__repr__()

    def getitem(self, *args, **kwargs):
        pass

    @VariableFactory.register_from_value()
    def from_value(
        value: Any,
        graph: FunctionGraph | None,
        tracker: Tracker,
    ):
        if isinstance(value, VariableBase):
            return value
        return None
