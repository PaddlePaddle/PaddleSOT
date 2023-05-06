from __future__ import annotations

from typing import TYPE_CHECKING, Any

import paddle

from ...infer_meta import MetaInfo
from ...proxy_tensor import ProxyTensor, ProxyTensorContext
from ...utils import NameGenerator
from ...utils.exceptions import InnerError
from .source import ConstSource, DummySource, GetItemSource, Source

if TYPE_CHECKING:
    from .function_graph import FunctionGraph


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


class VariableTrackerFactory:
    @staticmethod
    def from_value(
        value: Any, graph: FunctionGraph | None, source: Source | None
    ):
        if isinstance(value, VariableTracker):
            return value
        elif isinstance(value, (int, float, str, bool)):
            return ConstantVariable(value, source)
        elif isinstance(value, (paddle.Tensor, ProxyTensor)):
            assert graph is not None
            return TensorVariable(value, graph, source)
        elif isinstance(value, list):
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
        elif isinstance(value, tuple):
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
        raise RuntimeError(
            f"Don't Implement a value binding method for type: `{type(value)}`"
        )


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
        if not isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        retval = self._list[key.value]

        # if list is an input of funciton, we need make sure __getitem__ returns a VariableTracker
        if not isinstance(retval, VariableTracker):
            retval = VariableTrackerFactory.from_value(retval, GetItemSource(self, key))

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
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        if not isinstance(value, VariableTracker):
            raise InnerError(f"[{self.__class__.__name__}]: recieved {value} to set value.")

        self._list[key.value] = value

    def __delitem__(self, key):
        if not isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key to delete."
            )
        del self._list[key.value]


class TupleVariable(VariableTracker):
    def __init__(
        self, val_tuple: list[VariableTracker], source: Source | None
    ):
        super().__init__(source)
        self._tuple = val_tuple  

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
        if not isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        retval = self._tuple[key.value]

        if not isinstance(retval, VariableTracker):
            retval = VariableTrackerFactory.from_value(retval, GetItemSource(self, key))

        return retval

    def __setitem__(self, key, value):
        raise InnerError(f"[{self.__class__.__name__}]: setitem is not allowed.")

    def __delitem__(self, key):
        raise InnerError(f"[{self.__class__.__name__}]: delitem is not allowed.")


class DictVariable(VariableTracker):
    def __init__(
        self, val_dict: dict[VariableTracker: VariableTracker], source: Source | None
    ):
        super().__init__(source)
        self._dict = val_dict

    @staticmethod
    def build_from_vals(val_for_dict, source):
        new_dict = {}
        dict_len = len(val_for_dict)//2
        for i in range(dict_len):
            new_dict[val_for_dict[i]] = val_for_dict[i+1]
        return DictVariable(new_dict, source)

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

        if not isinstance(retval, VariableTracker):
            retval = VariableTrackerFactory.from_value(retval, GetItemSource(self, key))

        return retval

    def __setitem__(self, key, value):
        if not isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        if not isinstance(value, VariableTracker):
            raise InnerError(f"[{self.__class__.__name__}]: recieved {value} to set value.")

        self._dict[key.value] = value

    def __delitem__(self, key):
        if not isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key to delete."
            )
        del self._dict[key.value]