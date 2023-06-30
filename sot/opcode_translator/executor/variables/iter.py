from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..tracker import ConstTracker, DummyTracker, Tracker
from .base import VariableBase
from .basic import ConstantVariable, TensorVariable
from .container import DictVariable, ListVariable, RangeVariable, TupleVariable

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph


class IterVariable(VariableBase):
    def __init__(self, obj, graph, tracker):
        super().__init__(tracker)
        self.hold = obj
        self.graph = graph

    def next(self):
        raise NotImplementedError("")


class SequenceIterVariable(IterVariable):
    def __init__(self, obj, graph, tracker):
        super().__init__(obj, graph, tracker)
        self.idx = 0

    def next(self):
        if self.idx < len(self.hold):
            val = self.hold[self.idx]
            self.idx += 1
            return val
        else:
            raise StopIteration()

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
        }


class EnumerateVariable(IterVariable):
    # TODO(zmh): modify comments
    """
    EnumerateVariable is a subclass of IterVariable used to wrap a Variable of the enumerate type.

    Args:
        val_iterator (Iterable): The Iterable to be wrapped.
        graph (FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker (Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(self, val_iterator, graph, tracker):
        super().__init__(val_iterator, graph, tracker)
        self.idx = 0

    def next(self):
        if self.idx < len(self.hold):
            val = self.hold[self.idx]
            # wrap
            idx_var = ConstantVariable(
                self.idx, self.graph, ConstTracker(self.idx)
            )
            self.idx += 1
            return TupleVariable(
                (idx_var, val), self.graph, DummyTracker([idx_var, val])
            )
        else:
            raise StopIteration()

    # TODO(zmh): 添加其他方法
    @staticmethod
    def from_iterator(value, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(
            value, (ListVariable, TupleVariable, RangeVariable, DictVariable)
        ):
            return EnumerateVariable(value, graph, tracker)
        elif isinstance(value, TensorVariable):
            return TensorIterVariable(value, graph, tracker)
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


class TensorIterVariable(IterVariable):
    def __init__(self, obj, graph, tracker):
        super().__init__(obj, graph, tracker)


# what UserDefinedIterVariable holds doesn't matter, because use user defined iterator will trigger break graph
class UserDefinedIterVariable(IterVariable):
    def __init__(self, obj, graph, tracker):
        super().__init__(obj, graph, tracker)
