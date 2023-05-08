from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, Callable

import paddle

from ...infer_meta import MetaInfo
from ...proxy_tensor import ProxyTensor, ProxyTensorContext
from ...utils import NameGenerator
from ...utils.exceptions import InnerError
from .source import ConstSource, DummySource, GetItemSource, Source

if TYPE_CHECKING:
    from .function_graph import FunctionGraph


class VariableTrackerFactory:
    registered_funcs: list[Callable] = []

    @staticmethod
    def default_from_value(value, graph, source):
        return value
        raise RuntimeError(
            f"Don't Implement a value binding method for type: `{type(value)}`"
        )

    @staticmethod
    def register_from_value(from_value_func: Callable):
        VariableTrackerFactory.registered_funcs.append(from_value_func)

    @staticmethod
    def from_value(
        value: Any, graph: FunctionGraph | None, source: Source | None
    ):
        for func in VariableTrackerFactory.registered_funcs:
            var = func(value, graph, source)
            if var is not None:
                return var
        return VariableTrackerFactory.default_from_value(value, graph, source)


class VariableTracker:
    """
    we first deal guard information collection.
    """

    source: Source
    name_generator = NameGenerator("tracker_")

    def __init__(self, source: Source | None = None):
        if source is not None:
            self.source = source
        else:
            self.source = DummySource()
        self.id = VariableTracker.name_generator.next()

    def make_check_fn(self):
        raise NotImplementedError()

    def call_function(self, *args, **kwargs):
        pass

    def getattr(self, *args, **kwargs):
        pass

    def getitem(self, *args, **kwargs):
        pass

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any, graph: FunctionGraph | None, source: Source | None
    ):
        if isinstance(value, VariableTracker):
            return value
        return None


class ConstantVariable(VariableTracker):
    def __init__(self, value: Any, source: Source | None):
        super().__init__(source)
        self.value = value

    def make_check_fn(self):
        def guard_fn(frame):
            value = self.source.trace_value_from_frame()(frame)
            return isinstance(value, type(self.value)) and value == self.value

        return guard_fn

    def __repr__(self) -> str:
        return f"ConstantVariable({self.value})"

    def __mul__(self, other):
        if not isinstance(other, ConstantVariable):
            return NotImplemented
        var = VariableTrackerFactory.from_value(
            self.value * other.value, None, None
        )
        return var

    def __add__(self, other):
        if not isinstance(other, ConstantVariable):
            return NotImplemented
        var = VariableTrackerFactory.from_value(
            self.value + other.value, None, None
        )
        return var

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any, graph: FunctionGraph | None, source: Source | None
    ):
        if isinstance(value, (int, float, str, bool, type(None))):
            return ConstantVariable(value, source)
        return None


class TensorVariable(VariableTracker):
    def __init__(self, tensor, graph: FunctionGraph, source: Source | None):
        super().__init__(source)
        self.leaf = False
        if isinstance(tensor, (paddle.Tensor)):
            self.value = ProxyTensorContext().from_tensor(tensor)
            self.leaf = True
        elif isinstance(tensor, ProxyTensor):
            self.value = tensor
        self.graph = graph

    def make_check_fn(self):
        def guard_fn(frame):
            value = self.source.trace_value_from_frame()(frame)
            return (
                isinstance(value, paddle.Tensor)
                and MetaInfo.from_tensor(value) == self.value.meta
            )

        return guard_fn

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

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any, graph: FunctionGraph | None, source: Source | None
    ):
        if isinstance(value, (paddle.Tensor, ProxyTensor)):
            assert graph is not None
            return TensorVariable(value, graph, source)
        return None


class ListVariable(VariableTracker):
    def __init__(self, val_list: list[VariableTracker], source: Source | None):
        super().__init__(source)
        # everything in stack is VariableTracker, so just accept the input list is ok
        self._list = val_list

    def make_check_fn(self):
        def guard_fn(frame):
            for var in self._list:
                if not var.make_check_fn()(frame):
                    return False
            return True

        return guard_fn

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

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any, graph: FunctionGraph | None, source: Source | None
    ):
        if isinstance(value, list):
            return ListVariable(
                [
                    VariableTrackerFactory.from_value(
                        item,
                        graph,
                        GetItemSource(source, ConstSource(i))
                        if source is not None
                        else None,
                    )
                    for i, item in enumerate(value)
                ],
                source,
            )
        return None


class TupleVariable(VariableTracker):
    def __init__(
        self, val_tuple: tuple[VariableTracker], source: Source | None
    ):
        super().__init__(source)
        self._tuple = val_tuple  # exactly it is a list
        # (need replace item with VaraibleTracker)

    def make_check_fn(self):
        def guard_fn(frame):
            for var in self._tuple:
                if not var.make_check_fn()(frame):
                    return False
            return True

        return guard_fn

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

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any, graph: FunctionGraph | None, source: Source | None
    ):
        if isinstance(value, tuple):
            return TupleVariable(
                tuple(
                    VariableTrackerFactory.from_value(
                        item,
                        graph,
                        GetItemSource(source, ConstSource(i))
                        if source is not None
                        else None,
                    )
                    for i, item in enumerate(value)
                ),
                source,
            )
        return


class FunctionVariable(VariableTracker):
    def __init__(self, func, source):
        super().__init__(source)
        self.value = func

    def __call__(self, *args, **kwargs):
        pass

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any, graph: FunctionGraph | None, source: Source | None
    ):
        if isinstance(value, (types.FunctionType)):
            return FunctionVariable(value, source)
        return None
