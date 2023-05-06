from __future__ import annotations

import types
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable

import paddle

from ...infer_meta import MetaInfo
from ...proxy_tensor import ProxyTensor, ProxyTensorContext
from ...utils import NameGenerator, log_do
from ...utils.exceptions import InnerError
from .tracker import DummyTracker, GetItemTracker, Tracker

if TYPE_CHECKING:
    from .function_graph import FunctionGraph

Guard = Callable[[types.FrameType], bool]


def compose_guards(guards: list[Guard]) -> Guard:
    def composed_guard_fn(frame: types.FrameType) -> bool:
        ret = True
        for guard in guards:
            ret = ret and guard(frame)
        return ret

    return composed_guard_fn


def get_zero_degree_vars(
    variables: set[VariableTracker], visited_vars: list[VariableTracker]
) -> list[VariableTracker]:
    return [
        var
        for var in variables
        if var not in visited_vars
        and len(set(var.get_inputs()) - set(visited_vars)) == 0
    ]


def topo_sort_vars(
    root_variables: list[VariableTracker],
) -> list[VariableTracker]:
    variables = set()
    for root in root_variables:
        variables.add(root)
        variables |= set(root.flatten_inputs())

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

    tracker: Tracker
    name_generator = NameGenerator("tracker_")

    def __init__(self, tracker: Tracker):
        self.tracker = tracker
        self.id = VariableTracker.name_generator.next()

    def make_check_fn(self) -> Guard:
        assert not isinstance(
            self.tracker, DummyTracker
        ), "Can not make guard from dummy source"

        def guard_fn(frame: types.FrameType) -> bool:
            value = self.tracker.trace_value_from_frame()(frame)
            log_do(
                3,
                lambda: print(
                    f"[Guard]: guard_fn for {self}, tracker={self.tracker.__class__.__name__}, value={value}"
                ),
            )
            if isinstance(self, TensorVariable):
                return MetaInfo.from_tensor(value) == self.get_value().meta
            return self.get_value() == value

        return guard_fn

    def get_value(self) -> Any:
        raise NotImplementedError()

    def get_inputs(self) -> list[VariableTracker]:
        return self.tracker.inputs

    def flatten_inputs(self) -> list[VariableTracker]:
        flattened_inputs = []
        for input in self.get_inputs():
            flattened_inputs.extend(input.flatten_inputs())
        flattened_inputs.append(self)
        return flattened_inputs

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
        tracker: Tracker,
    ):
        if isinstance(value, VariableTracker):
            return value
        elif isinstance(value, (int, float, str, bool, type(None))):
            return ConstantVariable(value, tracker=tracker)
        elif isinstance(value, (paddle.Tensor, ProxyTensor)):
            assert graph is not None
            return TensorVariable(value, graph, tracker=tracker)
        elif isinstance(value, list):
            assert graph is not None
            return ListVariable(value, graph=graph, tracker=tracker)
        elif isinstance(value, tuple):
            assert graph is not None
            return TupleVariable(list(value), graph=graph, tracker=tracker)
        return
        raise RuntimeError(
            f"Don't Implement a value binding method for type: `{type(value)}`"
        )


class ConstantVariable(VariableTracker):
    def __init__(
        self,
        value: Any,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.value = value

    def get_value(self):
        return self.value

    def __repr__(self) -> str:
        return f"ConstantVariable({self.value})"

    def __mul__(self, other):
        if not isinstance(other, ConstantVariable):
            return NotImplemented
        var = VariableTrackerFactory.from_value(
            self.value * other.value, None, tracker=DummyTracker([self, other])
        )
        return var

    def __add__(self, other):
        if not isinstance(other, ConstantVariable):
            return NotImplemented
        var = VariableTrackerFactory.from_value(
            self.value + other.value, None, tracker=DummyTracker([self, other])
        )
        return var


class TensorVariable(VariableTracker):
    def __init__(
        self,
        tensor,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.leaf = False
        if isinstance(tensor, (paddle.Tensor)):
            self.value = ProxyTensorContext().from_tensor(tensor)
            self.leaf = True
        elif isinstance(tensor, ProxyTensor):
            self.value = tensor
        self.graph = graph

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
    def __init__(
        self,
        val_list: list[VariableTracker],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.graph = graph
        # everything in stack is VariableTracker, so just accept the input list is ok
        self._list = val_list

    def get_value(self):
        return self._list

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
            assert isinstance(key.value, int)
            index = key.value
        except:
            raise InnerError(
                "[ListVariable]: received {key}:{key.value} as key."
            )

        realval = self._list[index]
        return VariableTrackerFactory.from_value(
            realval, self.graph, tracker=GetItemTracker(self, key)
        )

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
                "[ListVariable]: received {key}:{key.value} as key."
            )

        if not isinstance(value, VariableTracker):
            raise InnerError("[ListVariable]: received {value} to set value.")

    def __delitem__(self, key):
        try:
            assert isinstance(key, ConstantVariable)
            index = int(key.value)
        except:
            raise InnerError(
                "[ListVariable]: received {key}:{key.value} as key."
            )

        del self._list[index]


class TupleVariable(VariableTracker):
    def __init__(
        self,
        val_tuple: list[VariableTracker],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.graph = graph
        # exactly it is a list (need replace item with VariableTracker)
        self._tuple = val_tuple

    def get_value(self):
        return tuple(self._tuple)

    def __repr__(self) -> str:
        return f"TupleVariable(len={len(self)})"

    def __len__(self):
        return len(self._tuple)

    def __getitem__(self, key):
        try:
            assert isinstance(key, ConstantVariable)
            assert isinstance(key.value, int)
            index = key.value
        except:
            raise InnerError(
                "[TupleVariable]: received {key}:{key.value} as key."
            )

        realval = self._tuple[index]
        return VariableTrackerFactory.from_value(
            realval, self.graph, tracker=GetItemTracker(self, key)
        )

    def __setitem__(self, key, value):
        raise InnerError("[TupleVariable]: setitem is not allowed.")

    def __delitem__(self, key):
        raise InnerError("[TupleVariable]: delitem is not allowed.")
