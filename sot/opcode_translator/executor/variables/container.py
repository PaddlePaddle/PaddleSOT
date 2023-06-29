from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

from ....utils import log_do
from ....utils.exceptions import InnerError, NotImplementException
from ..guard import StringifyExpression
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
        return reduce(
            operator.and_,
            [len_guard]
            + [item.make_stringify_guard() for item in self.get_items()],
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
        return ConstantVariable.wrap_literal(None)

    def __delitem__(self, key):
        return self.delitem(key)

    def delitem(self, key):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} as key to delete."
            )
        del self.value[key]
        return ConstantVariable.wrap_literal(None)

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


class RangeVariable(ContainerVariable):
    def __init__(
        self,
        val_range: range,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.graph = graph
        self.value = val_range

    def get_type(self):
        return range

    def get_value(self):
        return self.value

    def getitem(self, key):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        retval = self.value[key]
        return ConstantVariable.wrap_literal(retval)

    def get_items(self):
        size = len(self)
        return [self[idx] for idx in range(size)]

    def get_wrapped_items(self):
        return self.get_items()

    def __len__(self):
        return len(self.value)

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_global("range")
        # The start default value is 0, step is 1
        # So we can always construct range with 3 args
        codegen.gen_load_const(self.value.start)
        codegen.gen_load_const(self.value.stop)
        codegen.gen_load_const(self.value.step)
        codegen.gen_call_function(3)

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, range):
            return RangeVariable(value, graph, tracker)
        return None

    @property
    def debug_name(self) -> str:
        return ":".join(
            [
                str(self.value.start) if self.value.start is not None else "",
                str(self.value.stop) if self.value.stop is not None else "",
                str(self.value.step) if self.value.step is not None else "",
            ]
        )

    @debug_name.setter
    def debug_name(self, name):
        pass

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}


class DictVariable(ContainerVariable):
    def __init__(
        self,
        val_dict: dict[object, VariableBase],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.graph = graph
        self.value = val_dict

    def get_value(self):
        return {key: self[key].get_value() for key in self.value}

    def get_type(self):
        return dict

    def _reconstruct(self, codegen: PyCodeGen):
        from .basic import ConstantVariable

        size = len(self)
        for key in self.value.keys():
            if not isinstance(key, ConstTypes):
                raise InnerError(
                    f"[{self.__class__.__name__}]: recieved {key} as key."
                )
            key_var = ConstantVariable.wrap_literal(key)
            value_var = self[key]
            key_var.reconstruct(codegen)
            value_var.reconstruct(codegen)
        codegen.gen_build_map(size)

    def get_items(self):
        items = []
        for key in self.value.keys():
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
        for key in self.value.keys():
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
        return len(self.value)

    def getitem(self, key):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        retval = self.value[key]

        return VariableFactory.from_value(
            retval, self.graph, tracker=GetItemTracker(self, key)
        )

    def setitem(self, key, value):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        if not isinstance(value, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {value} to set value."
            )

        self.value[key] = value

        return ConstantVariable.wrap_literal(None)

    def __delitem__(self, key):
        return self.delitem(key)

    def delitem(self, key):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key to delete."
            )
        del self.value[key]
        return ConstantVariable.wrap_literal(None)

    def keys(self):
        from .iter import SequenceIterVariable

        raw_list = [
            ConstantVariable(x, ConstTracker(x)) for x in self.value.keys()
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

        keys = [ConstantVariable(x, ConstTracker(x)) for x in self.value.keys()]
        values = list(self.get_wrapped_items().values())
        raw_list = list(zip(keys, values))
        item_list = VariableFactory.from_value(
            raw_list, self.graph, DummyTracker([self])
        )
        return SequenceIterVariable(
            item_list, self.graph, DummyTracker([item_list])
        )

    def update(self, data):
        self.value.update(data.get_wrapped_items())
        return self

    def getattr(self, name):
        from .callable import BuiltinVariable

        method_name_to_builtin_fn = {
            "keys": dict.keys,
            "values": dict.values,
            "items": dict.items,
            "update": dict.update,
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
