from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

from ....utils import log_do
from ....utils.exceptions import InnerError, NotImplementException
from ..guard import StringifyExpression
from ..mutable_data import MutableDictLikeData
from ..pycode_generator import PyCodeGen
from ..tracker import (
    ConstTracker,
    DanglingTracker,
    DummyTracker,
    GetItemTracker,
    Tracker,
)
from .base import ConstTypes, VariableBase, VariableFactory
from .basic import ConstantVariable

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph


class ContainerVariable(VariableBase):
    def get_items(self) -> list[VariableBase]:
        raise NotImplementException()

    def __len__(self):
        raise NotImplementException()

    def len(self):
        return VariableFactory.from_value(
            len(self), self.graph, DummyTracker([self])
        )

    def __bool__(self) -> bool:
        return len(self) > 0

    def bool(self):
        return VariableFactory.from_value(
            bool(self), self.graph, DummyTracker([self])
        )

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
        len_guard = StringifyExpression(
            f"len({frame_value_tracer.expr}) == {len(self)}",
            frame_value_tracer.free_vars,
        )
        guard_variables = filter(
            lambda var: var.tracker.is_traceable(), self.get_items()
        )
        return reduce(
            operator.and_,
            [len_guard]
            + [item.make_stringify_guard() for item in guard_variables],
        )


class ListVariable(ContainerVariable):
    def __init__(
        self,
        val_list: list[VariableBase],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.graph = graph
        # everything in stack is VariableBase, so just accept the input list is ok
        self.value = val_list

    def get_value(self):
        return [self[idx].get_value() for idx in range(len(self))]

    def get_type(self):
        return list

    def _reconstruct(self, codegen: PyCodeGen):
        self.graph.add_global_guarded_variable(self)
        size = len(self)
        for idx in range(size):
            self[idx].reconstruct(codegen)
        codegen.gen_build_list(size)

    def get_items(self):
        size = len(self)
        return [self[idx] for idx in range(size)]

    def get_wrapped_items(self):
        return self.get_items()

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "len": len(self),
        }

    def __len__(self):
        return len(self.value)

    def getitem(self, key):
        '''
        we need to make sure that:
            before an inplace change happens to ListVariable,
            the related items should already be wrapped as VariableBase

        if not, tracker might be set to a wrong elem
        '''
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        retval = self.value[key]

        # if list is an input of funciton, we need make sure __getitem__ returns a VariableBase
        retval = VariableFactory.from_value(
            retval, self.graph, tracker=GetItemTracker(self, key)
        )

        return retval

    def setitem(self, key, value):
        '''
        why setitem is ok:

        case:
            def f(x = [t0, t1])
                ...
                x[0] = 0
                ...

            1. if setitem happens after get t0: t0 is a VariableBase (transformed at getitem), so it is ok
            2. if setitem happens before get t0: t0 will not be used
        '''
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} as key."
            )

        if not isinstance(value, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {value} to set value."
            )
        self.value[key] = value
        return ConstantVariable.wrap_literal(None, self.graph)

    def __delitem__(self, key):
        return self.delitem(key)

    def delitem(self, key):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} as key to delete."
            )
        del self.value[key]
        return ConstantVariable.wrap_literal(None, self.graph)

    def extend(self, data):
        self.value.extend(data.get_wrapped_items())
        return self

    def concat(self, list_):
        assert isinstance(list_, ListVariable)
        new_list_variable = ListVariable(
            self.get_wrapped_items() + list_.get_wrapped_items(),
            self.graph,
            DummyTracker([self, list_]),
        )
        return new_list_variable

    def repeat(self, length):
        assert isinstance(length, ConstantVariable)
        new_list_variable = ListVariable(
            self.get_wrapped_items() * length.value,
            self.graph,
            DummyTracker([self, length]),
        )
        return new_list_variable

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, list):
            assert graph is not None
            return ListVariable(value, graph=graph, tracker=tracker)
        return None


class TupleVariable(ContainerVariable):
    def __init__(
        self,
        val_tuple: tuple[VariableBase],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.graph = graph
        self.value = val_tuple

    def get_value(self):
        return tuple(self[idx].get_value() for idx in range(len(self)))

    def get_type(self):
        return tuple

    def _reconstruct(self, codegen: PyCodeGen):
        self.graph.add_global_guarded_variable(self)
        size = len(self)
        for idx in range(size):
            self[idx].reconstruct(codegen)
        codegen.gen_build_tuple(size)

    def get_items(self):
        size = len(self)
        return [self[idx] for idx in range(size)]

    def get_wrapped_items(self):
        return self.get_items()

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "len": len(self),
        }

    def __len__(self):
        return len(self.value)

    def getitem(self, key):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )
        retval = self.value[key]

        return VariableFactory.from_value(
            retval, graph=self.graph, tracker=GetItemTracker(self, key)
        )

    def setitem(self, key, value):
        raise InnerError(
            f"[{self.__class__.__name__}]: setitem is not allowed."
        )

    def __delitem__(self, key):
        return self.delitem(key)

    def delitem(self, key):
        raise InnerError(
            f"[{self.__class__.__name__}]: delitem is not allowed."
        )

    def concat(self, tuple_):
        assert isinstance(tuple_, TupleVariable)
        new_tuple_variable = TupleVariable(
            self.get_wrapped_items() + tuple_.get_wrapped_items(),
            self.graph,
            DummyTracker([self, tuple_]),
        )
        return new_tuple_variable

    def repeat(self, length):
        assert isinstance(length, ConstantVariable)
        new_tuple_variable = TupleVariable(
            self.get_wrapped_items() * length.value,
            self.graph,
            DummyTracker([self, length]),
        )
        return new_tuple_variable

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, tuple):
            return TupleVariable(value, graph, tracker)
        return None


class DictVariable(ContainerVariable):
    def __init__(
        self,
        val_dict: dict[object, VariableBase],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.graph = graph
        self.proxy = self.graph.side_effects.get_proxy(
            MutableDictLikeData, val_dict, self.proxy_getter
        )
        self.value = val_dict

    def proxy_getter(self, data, key):
        if key not in data:
            return MutableDictLikeData.Empty()
        return VariableFactory.from_value(
            data[key], self.graph, tracker=GetItemTracker(self, key)
        )

    def get_value(self):
        return {
            key: value.get_value()
            for key, value in self.proxy.get_all().items()
        }

    def get_type(self):
        return dict

    def _reconstruct(self, codegen: PyCodeGen):
        from .basic import ConstantVariable

        self.graph.add_global_guarded_variable(self)

        size = len(self)
        for key in self.proxy.get_all().keys():
            if not isinstance(key, ConstTypes):
                raise InnerError(
                    f"[{self.__class__.__name__}]: recieved {key} as key."
                )
            key_var = ConstantVariable.wrap_literal(key, self.graph)
            value_var = self[key]
            key_var.reconstruct(codegen)
            value_var.reconstruct(codegen)
        codegen.gen_build_map(size)

    def get_items(self):
        items = []
        for key in self.proxy.get_all().keys():
            if not isinstance(key, ConstTypes):
                raise InnerError(
                    f"[{self.__class__.__name__}]: recieved {key} as key."
                )
            key_var = VariableFactory.from_value(
                key, self.graph, tracker=ConstTracker(key)
            )
            value_var = self[key]
            items.extend([key_var, value_var])
        return items

    def get_wrapped_items(self):
        items = {}
        for key in self.proxy.get_all().keys():
            if not isinstance(key, ConstTypes):
                raise InnerError(
                    f"[{self.__class__.__name__}]: recieved {key} as key."
                )
            items[key] = self[key]
        return items

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "len": len(self),
        }

    def __len__(self):
        return len(self.proxy.get_all())

    def get(self, key, default=None):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} to get value."
            )

        if default is None:
            return self.getitem(key)

        if isinstance(self.proxy.get(key), MutableDictLikeData.Empty):
            if isinstance(default, VariableBase):
                return default
            return VariableFactory.from_value(default)

        return self.getitem(key)

    def getitem(self, key):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        return self.proxy.get(key)

    def setitem(self, key, value):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        if not isinstance(value, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {value} to set value."
            )

        self.proxy.set(key, value)
        self.graph.side_effects.record_variable(self)

        return ConstantVariable.wrap_literal(None, self.graph)

    def clear(self):
        # TODO: Replace with self.proxy.clear()
        for key in self.value:
            self.delitem(key)
        self.graph.side_effects.record_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def __delitem__(self, key):
        return self.delitem(key)

    def delitem(self, key):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key to delete."
            )
        self.proxy.delete(key)
        self.graph.side_effects.record_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def keys(self):
        from .iter import SequenceIterVariable

        raw_list = [
            ConstantVariable(x, self.graph, ConstTracker(x))
            for x in self.proxy.get_all().keys()
        ]
        key_list = VariableFactory.from_value(
            raw_list, self.graph, ConstTracker(raw_list)
        )
        return SequenceIterVariable(
            key_list, self.graph, DummyTracker([key_list])
        )

    def values(self):
        from .iter import SequenceIterVariable

        raw_list = list(self.get_wrapped_items().values())
        value_list = VariableFactory.from_value(
            raw_list, self.graph, DummyTracker([self])
        )
        return SequenceIterVariable(
            value_list, self.graph, DummyTracker([value_list])
        )

    def items(self):
        from .iter import SequenceIterVariable

        keys = [
            ConstantVariable(x, self.graph, ConstTracker(x))
            for x in self.proxy.get_all().keys()
        ]
        values = list(self.get_wrapped_items().values())
        raw_list = list(zip(keys, values))
        item_list = VariableFactory.from_value(
            raw_list, self.graph, DummyTracker([self])
        )
        return SequenceIterVariable(
            item_list, self.graph, DummyTracker([item_list])
        )

    def update(self, data: DictVariable):
        for key, value in data.proxy.get_all().items():
            self.setitem(key, value)
        return ConstantVariable.wrap_literal(None, self.graph)

    def copy(self):
        new_dict_variable = DictVariable(
            self.get_wrapped_items(), self.graph, DummyTracker([self])
        )
        return new_dict_variable

    def setdefault(self, key, default=None):
        if isinstance(self.proxy.get(key), MutableDictLikeData.Empty):
            if default is None:
                self.setitem(
                    key, ConstantVariable.wrap_literal(default, self.graph)
                )
            else:
                self.setitem(key, default)

        return self.getitem(key)

    def pop(self, key, default=None):
        if isinstance(self.proxy.get(key), MutableDictLikeData.Empty):
            if isinstance(default, VariableBase):
                return default
            return VariableFactory.from_value(default)

        # default is not None, or key is in dict
        temp_value = self.getitem(key)
        self.delitem(key)
        return temp_value

    def getattr(self, name):
        from .callable import BuiltinVariable

        method_name_to_builtin_fn = {
            "keys": dict.keys,
            "values": dict.values,
            "items": dict.items,
            "update": dict.update,
            "setdefault": dict.setdefault,
            "get": dict.get,
            "copy": dict.copy,
            "clear": dict.clear,
            "pop": dict.pop,
        }

        if name in method_name_to_builtin_fn:
            builtin_fn = method_name_to_builtin_fn[name]
            return BuiltinVariable(
                builtin_fn, self.graph, DanglingTracker()
            ).bind(self, name)
        else:
            raise NotImplementException(
                f"attribute {name} for dict is not implemented"
            )

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, dict):
            assert graph is not None
            return DictVariable(value, graph=graph, tracker=tracker)
