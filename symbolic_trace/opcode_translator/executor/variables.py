from __future__ import annotations

import types
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable

import paddle

from ...infer_meta import MetaInfo
from ...proxy_tensor import ProxyTensor, ProxyTensorContext
from ...utils import ASSERT, NameGenerator, log_do
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


class VariableTrackerFactory:
    registered_funcs: list[Callable] = []

    @staticmethod
    def default_from_value(value, graph, tracker):
        return value
        raise RuntimeError(
            f"Don't Implement a value binding method for type: `{type(value)}`"
        )

    @staticmethod
    def register_from_value(from_value_func: Callable):
        VariableTrackerFactory.registered_funcs.append(from_value_func)

    @staticmethod
    def from_value(
        value: Any, graph: FunctionGraph | None, tracker: Tracker | None
    ):
        for func in VariableTrackerFactory.registered_funcs:
            var = func(value, graph, tracker)
            if var is not None:
                return var
        return VariableTrackerFactory.default_from_value(value, graph, tracker)


class VariableTracker:
    """
    we first deal guard information collection.
    """

    tracker: Tracker
    name_generator = NameGenerator("tracker_")

    def __init__(self, tracker: Tracker):
        self.tracker = tracker
        self.id = VariableTracker.name_generator.next()

    def __hash__(self):
        return hash(self.id)

    def make_check_fn(self) -> Guard:
        assert not isinstance(
            self.tracker, DummyTracker
        ), "Can not make guard from dummy tracker"

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

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any,
        graph: FunctionGraph | None,
        tracker: Tracker,
    ):
        if isinstance(value, VariableTracker):
            return value
        return None


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

    def __sub__(self, other):
        if not isinstance(other, (ConstantVariable, TensorVariable)):
            return NotImplemented
        return self.graph.call_tensor_method("__sub__", self, other)

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any, graph: FunctionGraph | None, tracker: Tracker | None
    ):
        if isinstance(value, (int, float, str, bool, type(None))):
            return ConstantVariable(value, tracker)
        return None


class TensorVariable(VariableTracker):
    def __init__(
        self,
        tensor: paddle.Tensor | ProxyTensor,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.leaf = False
        if isinstance(tensor, paddle.Tensor):
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

    def __gt__(self, other):
        if not isinstance(other, (ConstantVariable, TensorVariable)):
            return NotImplemented
        return self.graph.call_tensor_method("__gt__", self, other)

    def __sub__(self, other):
        if not isinstance(other, (ConstantVariable, TensorVariable)):
            return NotImplemented
        return self.graph.call_tensor_method("__sub__", self, other)

    def __rsub__(self, other):
        if not isinstance(other, (ConstantVariable, TensorVariable)):
            return NotImplemented
        return self.graph.call_tensor_method("__rsub__", self, other)

    def __repr__(self) -> str:
        return f"TensorVariable{self.value.meta}"

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any, graph: FunctionGraph | None, tracker: Tracker | None
    ):
        if isinstance(value, (paddle.Tensor, ProxyTensor)):
            assert graph is not None
            return TensorVariable(value, graph, tracker)
        return None


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
        self._sequence = val_list

    def get_value(self):
        return self._sequence

    def __repr__(self) -> str:
        return f"ListVariable(len={len(self)})"

    def __len__(self):
        return len(self._sequence)

    def __getitem__(self, key):
        '''
        we need to make sure that:
            before an inplace change happens to ListVariable,
            the related items should already be wrapped as VariableTracker

        if not, tracker might be set to a wrong elem
        '''
        if not isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        retval = self._sequence[key.value]

        # if list is an input of funciton, we need make sure __getitem__ returns a VariableTracker
        retval = VariableTrackerFactory.from_value(
            retval, self.graph, tracker=GetItemTracker(self, key)
        )

        return retval

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
        if not isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} as key."
            )

        if not isinstance(value, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {value} to set value."
            )

        self._sequence[key.value] = value

    def __delitem__(self, key):
        if not isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} as key to delete."
            )
        del self._sequence[key.value]

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any, graph: FunctionGraph | None, tracker: Tracker | None
    ):
        if isinstance(value, list):
            assert graph is not None
            return ListVariable(value, graph=graph, tracker=tracker)
        return None


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
        self._sequence = val_tuple

    def get_value(self):
        return tuple(self._sequence)

    def __repr__(self) -> str:
        return f"TupleVariable(len={len(self)})"

    def __len__(self):
        return len(self._sequence)

    def __getitem__(self, key):
        if not isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )
        retval = self._sequence[key.value]
        return VariableTrackerFactory.from_value(
            retval, graph=self.graph, tracker=GetItemTracker(self, key)
        )

    def __setitem__(self, key, value):
        raise InnerError(
            f"[{self.__class__.__name__}]: setitem is not allowed."
        )

    def __delitem__(self, key):
        raise InnerError(
            f"[{self.__class__.__name__}]: delitem is not allowed."
        )

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any, graph: FunctionGraph | None, tracker: Tracker | None
    ):
        if isinstance(value, tuple):
            return TupleVariable(value, graph, tracker)
        return None


class DictVariable(VariableTracker):
    def __init__(
        self,
        val_dict: dict[object, VariableTracker],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.graph = graph
        self._dict = val_dict

    def __repr__(self) -> str:
        return f"DictVariable(len={len(self)})"

    def __len__(self):
        return len(self._dict)

    def __getitem__(self, key):
        if not isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        retval = self._dict[key.value]

        return VariableTrackerFactory.from_value(
            retval, self.graph, tracker=GetItemTracker(self, key)
        )

    def __setitem__(self, key, value):
        if not isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        if not isinstance(value, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {value} to set value."
            )

        self._dict[key.value] = value

    def __delitem__(self, key):
        if not isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key to delete."
            )
        del self._dict[key.value]

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any, graph: FunctionGraph | None, tracker: Tracker | None
    ):
        if isinstance(value, dict):
            assert graph is not None
            return DictVariable(value, graph=graph, tracker=tracker)


class FunctionVariable(VariableTracker):
    def __init__(self, func, graph, tracker):
        super().__init__(tracker)
        self.value = func
        self.graph = graph

    def __call__(self, *args, **kwargs):
        return self.call_function(*args, **kwargs)

    def call_function(self, *args, **kwargs):
        from .opcode_inline_executor import OpcodeInlineExecutor

        if self.value is ASSERT:
            return self.value(args[0].value)

        inline_executor = OpcodeInlineExecutor(self, *args, **kwargs)
        output = inline_executor.inline_call()
        return output

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any, graph: FunctionGraph | None, tracker: Tracker | None
    ):
        if isinstance(value, (types.FunctionType)):
            return FunctionVariable(value, graph, tracker)
        return None
