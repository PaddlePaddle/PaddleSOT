from __future__ import annotations

import types
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable

import paddle

from ...infer_meta import MetaInfo
from ...proxy_tensor import ProxyTensor, ProxyTensorContext
from ...utils import NameGenerator
from ...utils.exceptions import InnerError
from .source import ConstSource, DummySource, GetItemSource, Source

if TYPE_CHECKING:
    from .function_graph import FunctionGraph

Guard = Callable[[types.FrameType], bool]


def compose_guards(guards: list[Guard]) -> Guard:
    def composed_guard_fn(frame: types.FrameType) -> bool:
        for guard in guards:
            if not guard(frame):
                return False
        return True

    return composed_guard_fn


def get_zero_degree_vars(
    variables: set[VariableTracker], visited_vars: list[VariableTracker]
) -> list[VariableTracker]:
    return [
        var
        for var in variables
        if var not in visited_vars
        and len(set(var.deps) - set(visited_vars)) == 0
    ]


def topo_sort_vars(
    root_variables: list[VariableTracker],
) -> list[VariableTracker]:
    variables = set()
    for root in root_variables:
        variables.add(root)
        variables |= set(root.flatten_deps())

    topo_ordered_vars = []
    topo_queue = Queue()
    for var in get_zero_degree_vars(variables, topo_ordered_vars):
        topo_queue.put(var)

    while not topo_queue.empty():
        var = topo_queue.get()
        topo_ordered_vars.append(var)
        for zero_degree_var in get_zero_degree_vars(
            variables, topo_ordered_vars
        ):
            if (
                zero_degree_var in topo_queue.queue
                or zero_degree_var in topo_ordered_vars
            ):
                continue
            topo_queue.put(zero_degree_var)
    return topo_ordered_vars


class VariableTracker:
    """
    we first deal guard information collection.
    """

    source: Source
    deps: list[VariableTracker]
    check_value: bool
    name_generator = NameGenerator("tracker_")

    def __init__(self, source: Source | None = None, deps=[]):
        if source is not None:
            self.source = source
        else:
            self.source = DummySource()
        self.deps = deps
        self.id = VariableTracker.name_generator.next()

    def make_check_fn(self) -> Guard:
        assert not isinstance(
            self.source, DummySource
        ), "Can not make guard from dummy source"
        if not self.check_value:
            return lambda _: True

        def guard_fn(frame: types.FrameType) -> bool:
            value = self.source.trace_value_from_frame()(frame)
            if isinstance(self, TensorVariable):
                return MetaInfo.from_tensor(value) == self.get_value().meta
            return self.get_value() == value

        return guard_fn

    def flatten_deps(self):
        flattened_deps = []
        for dep in self.deps:
            flattened_deps.extend(dep.flatten_deps())
        flattened_deps.append(self)
        return flattened_deps

    def call_function(self, *args, **kwargs):
        pass

    def getattr(self, *args, **kwargs):
        pass

    def getitem(self, *args, **kwargs):
        pass


class VariableTrackerFactory:
    @staticmethod
    def from_value(
        value: Any,
        graph: FunctionGraph | None,
        source: Source | None,
        deps: list[VariableTracker] = [],
    ):
        if isinstance(value, VariableTracker):
            return value
        elif isinstance(value, (int, float, str, bool, type(None))):
            return ConstantVariable(value, source=source, deps=deps)
        elif isinstance(value, (paddle.Tensor, ProxyTensor)):
            assert graph is not None
            return TensorVariable(value, graph, source, deps=deps)
        elif isinstance(value, list):
            return ListVariable(
                [
                    VariableTrackerFactory.from_value(
                        item,
                        graph,
                        source=GetItemSource(source, ConstSource(i))
                        if source is not None
                        else None,
                        deps=deps,
                    )
                    for i, item in enumerate(value)
                ],
                source,
            )
        elif isinstance(value, tuple):
            return TupleVariable(
                tuple(
                    VariableTrackerFactory.from_value(
                        item,
                        graph,
                        source=GetItemSource(source, ConstSource(i))
                        if source is not None
                        else None,
                        deps=deps,
                    )
                    for i, item in enumerate(value)
                ),
                source,
            )
        return
        raise RuntimeError(
            f"Don't Implement a value binding method for type: `{type(value)}`"
        )


class ConstantVariable(VariableTracker):
    check_value: bool = True

    def __init__(
        self,
        value: Any,
        source: Source | None,
        deps: list[VariableTracker] = [],
    ):
        super().__init__(source, deps)
        self.value = value

    # def make_check_fn(self):
    #     def guard_fn(frame):
    #         value = self.source.trace_value_from_frame()(frame)
    #         return isinstance(value, type(self.value)) and value == self.value

    #     return guard_fn

    def get_value(self):
        return self.value

    def __repr__(self) -> str:
        return f"ConstantVariable({self.value})"

    def __mul__(self, other):
        if not isinstance(other, ConstantVariable):
            return NotImplemented
        var = VariableTrackerFactory.from_value(
            self.value * other.value, None, source=None, deps=[self, other]
        )
        return var

    def __add__(self, other):
        if not isinstance(other, ConstantVariable):
            return NotImplemented
        var = VariableTrackerFactory.from_value(
            self.value + other.value, None, source=None, deps=[self, other]
        )
        return var


class TensorVariable(VariableTracker):
    check_value: bool = True

    def __init__(
        self,
        tensor,
        graph: FunctionGraph,
        source: Source | None,
        deps: list[VariableTracker] = [],
    ):
        super().__init__(source, deps)
        self.leaf = False
        if isinstance(tensor, (paddle.Tensor)):
            self.value = ProxyTensorContext().from_tensor(tensor)
            self.leaf = True
        elif isinstance(tensor, ProxyTensor):
            self.value = tensor
        self.graph = graph

    # def make_check_fn(self):
    #     def guard_fn(frame):
    #         value = self.source.trace_value_from_frame()(frame)
    #         return (
    #             isinstance(value, paddle.Tensor)
    #             and MetaInfo.from_tensor(value) == self.value.meta
    #         )

    #     return guard_fn

    def get_value(self):
        return self.value

    def __rmul__(self, other):
        if not isinstance(other, (ConstantVariable, TensorVariable)):
            return NotImplemented
        return self.graph.call_tensor_method("__rmul__", self, other)

    def __mul__(self, other):
        if not isinstance(other, (ConstantVariable, TensorVariable)):
            return NotImplemented
        return self.graph.call_tensor_method("__mul__", self, other)

    def __add__(self, other):
        if not isinstance(other, (ConstantVariable, TensorVariable)):
            return NotImplemented
        return self.graph.call_tensor_method("__add__", self, other)

    def __radd__(self, other):
        if not isinstance(other, (ConstantVariable, TensorVariable)):
            return NotImplemented
        return self.graph.call_tensor_method("__radd__", self, other)

    def __repr__(self) -> str:
        return f"TensorVariable{self.value.meta}"


class ListVariable(VariableTracker):
    check_value: bool = False

    def __init__(
        self,
        val_list: list[VariableTracker],
        source: Source | None,
        deps: list[VariableTracker] = [],
    ):
        super().__init__(source, deps)
        # everything in stack is VariableTracker, so just accept the input list is ok
        self._list = val_list

    # def make_check_fn(self):
    #     def guard_fn(frame):
    #         for var in self._list:
    #             if not var.make_check_fn()(frame):
    #                 return False
    #         return True

    #     return guard_fn

    def __repr__(self) -> str:
        return f"ListVariable(len={len(self)})"

    def __len__(self):
        return len(self._list)

    def __getitem__(self, key):
        '''
        we need to make sure that:
            before an inplace change happens to ListVariable,
            the related items should already be wrapped as VariableTracker

        if not, source might be set to a wrong elem
        '''
        try:
            assert isinstance(key, ConstantVariable)
            index = int(key.value)
        except:
            raise InnerError(
                "[ListVariable]: recieved {key}:{key.value} as key."
            )

        return self._list[index]

    def __setitem__(self, key, value):
        '''
        why __setitem__ is ok:

        case:
            def f(x = [t0, t1])
                ...
                x[0] = 0
                ...

            1. if setitem happens after get t0: t0 is a VariableTracker (transformed at getitem), so it is ok
            2. if setitem happens before get t0: t0 will not be used
        '''
        try:
            assert isinstance(key, ConstantVariable)
            index = int(key.value)
        except:
            raise InnerError(
                "[ListVariable]: recieved {key}:{key.value} as key."
            )

        if not isinstance(value, VariableTracker):
            raise InnerError("[ListVariable]: recieved {value} to set value.")

        self._list[index] = value

    def __delitem__(self, key):
        try:
            assert isinstance(key, ConstantVariable)
            index = int(key.value)
        except:
            raise InnerError(
                "[ListVariable]: recieved {key}:{key.value} as key."
            )

        del self._list[index]


class TupleVariable(VariableTracker):
    check_value: bool = False

    def __init__(
        self,
        val_tuple: tuple[VariableTracker],
        source: Source | None,
        deps: list[VariableTracker] = [],
    ):
        super().__init__(source, deps)
        self._tuple = val_tuple  # exactly it is a list
        # (need replace item with VaraibleTracker)

    def __repr__(self) -> str:
        return f"TupleVariable(len={len(self)})"

    def __len__(self):
        return len(self._tuple)

    def __getitem__(self, key):
        try:
            assert isinstance(key, ConstantVariable)
            index = int(key.value)
        except:
            raise InnerError(
                "[TupleVariable]: recieved {key}:{key.value} as key."
            )

        return self._tuple[index]

    def __setitem__(self, key, value):
        raise InnerError("[TupleVariable]: setitem is not allowed.")

    def __delitem__(self, key):
        raise InnerError("[TupleVariable]: delitem is not allowed.")
