from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ....utils import NotImplementException
from ..pycode_generator import PyCodeGen
from ..tracker import ConstTracker, DummyTracker, GetIterTracker
from .base import VariableBase
from .basic import ConstantVariable, TensorVariable
from .container import DictVariable, ListVariable, RangeVariable, TupleVariable

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph
    from ..tracker import Tracker


class IterVariable(VariableBase):
    """
    This Variable (include subclasses) should be generated only when simulate GET_ITER opcode
    """

    def __init__(
        self, obj: VariableBase, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(graph, tracker)
        self.hold = obj

    def make_stringify_guard(self):
        return self.hold.make_stringify_guard()

    def next(self):
        raise NotImplementedError(f"Can not simulate `next` for {type(self)}")


class SequenceIterVariable(IterVariable):
    mutable_attrs = ["idx"]

    def __init__(self, obj, graph: FunctionGraph, tracker: Tracker):
        super().__init__(obj, graph, tracker)
        self.idx = 0
        self.graph.side_effects.record_mutable_variable(self)

    def next(self):
        # TODO: self.hold should have a __len__ method
        if self.idx < len(self.hold):
            val = self.hold[self.idx]
            self.idx += 1
            return val
        else:
            raise StopIteration()

    def to_list(self) -> list:
        if self.has_side_effect():
            raise NotImplementException(
                "Can not convert an used iterator into list"
            )
        self.idx = len(self.hold)
        return list(self.hold)

    def has_side_effect(self) -> bool:
        return self.idx != 0

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
        }

    def _reconstruct(self, codegen: PyCodeGen):
        if self.has_side_effect():
            super()._reconstruct(codegen)
        else:
            self.hold.reconstruct(codegen)
            codegen.gen_get_iter()


class EnumerateVariable(IterVariable):

    """
    EnumerateVariable is a subclass of IterVariable used to wrap an Iteraable type.

    Args:
        val_iterator (Iterable): The Iterable to be wrapped.
        graph (FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker (Tracker): The Tracker object that tracks the information of this variable.
    """

    mutable_attrs = ["idx"]

    def __init__(self, val_iterator, graph, tracker):
        super().__init__(val_iterator, graph, tracker)
        self.idx = 0
        self.graph.side_effects.record_mutable_variable(self)

    def next(self):
        val = self.hold.next()
        idx_var = ConstantVariable(self.idx, self.graph, ConstTracker(self.idx))
        self.idx += 1
        return TupleVariable(
            (idx_var, val), self.graph, DummyTracker([idx_var, val])
        )

    def has_side_effect(self) -> bool:
        return self.hold.has_side_effect() or self.idx != 0

    def _reconstruct(self, codegen: PyCodeGen):
        if self.has_side_effect():
            super()._reconstruct(codegen)
        else:
            codegen.gen_load_global("enumerate")
            self.hold.reconstruct(codegen)
            self.gen_call_function(1)

    def to_list(self):
        values = self.hold.to_list()
        idx = [
            ConstantVariable(i, self.graph, ConstTracker(i))
            for i in range(len(values))
        ]
        return list(zip(idx, values))

    @staticmethod
    def from_iterator(value, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(
            value,
            (
                ListVariable,
                TupleVariable,
                RangeVariable,
            ),
        ):
            inner_iter = SequenceIterVariable(
                value, graph, GetIterTracker(value)
            )
            return EnumerateVariable(inner_iter, graph, tracker)

        elif isinstance(value, DictVariable):
            inner_iter = value.keys()
            return EnumerateVariable(inner_iter, graph, tracker)

        elif isinstance(value, TensorVariable):
            inner_iter = TensorIterVariable(value, graph, GetIterTracker(value))
            return EnumerateVariable(inner_iter, graph, tracker)

        elif isinstance(value, SequenceIterVariable):
            return EnumerateVariable(value, graph, tracker)

        else:
            return UserDefinedIterVariable(value, graph, tracker)


class DictIterVariable(IterVariable):
    def __init__(self, obj, graph, tracker):
        super().__init__(obj, graph, tracker)
        self.key_list = [
            ConstantVariable(x, graph, ConstTracker(x)) for x in self.hold
        ]
        self.idx = 0

    def next(self):
        if self.idx < len(self.key_list):
            val = self.key_list[self.idx]
            return val
        else:
            raise StopIteration()


class TensorIterVariable(SequenceIterVariable):
    def __init__(self, obj, graph, tracker):
        super().__init__(obj, graph, tracker)

    def to_list(self) -> list:
        if self.has_side_effect():
            raise NotImplementException(
                "Can not convert an used iterator into list"
            )
        self.idx = len(self.hold)
        retval = []
        for i in range(len(self.hold)):
            retval.append(self.hold[i])
        return retval


# what UserDefinedIterVariable holds doesn't matter, because use user defined iterator will trigger break graph
class UserDefinedIterVariable(IterVariable):
    def __init__(self, obj, graph, tracker):
        super().__init__(obj, graph, tracker)
