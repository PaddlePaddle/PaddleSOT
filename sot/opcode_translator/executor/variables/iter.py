from __future__ import annotations

from typing import TYPE_CHECKING

from ..tracker import ConstTracker
from .base import VariableBase
from .basic import ConstantVariable

if TYPE_CHECKING:
    pass


class IterVariable(VariableBase):
    def __init__(self, obj, graph, tracker):
        super().__init__(tracker)
        self.hold = obj
        self.graph = graph


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

    def __repr__(self):
        return f"SequenceIterVariable(idx={self.idx})"


class DictIterVariable(IterVariable):
    def __init__(self, obj, graph, tracker):
        super().__init__(obj, graph, tracker)
        self.key_list = [
            ConstantVariable(x, ConstTracker(x)) for x in self.hold
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
