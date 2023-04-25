import paddle

from ..proxy_tensor import ProxyTensor, ProxyTensorContext, callable_wrapper
from ..utils import NameGenerator


class VariableTracker:
    """
    we first deal guard information collection.
    """

    name_generator = NameGenerator("tracker_")

    def __init__(self):
        self.source = None
        self.id = VariableTracker.name_generator.next()
        pass

    def set_source(self, source):
        self.source = source

    def make_check_fn(self):
        self.source.gen_guard_fn(self.value)

    def call_function(self, *args, **kwargs):
        pass

    def getattr(self, *args, **kwargs):
        pass

    def getitem(self, *args, **kwargs):
        pass


class VariableTrackerFactory:
    @staticmethod
    def from_value(value, graph):
        if isinstance(value, VariableTracker):
            return value
        elif isinstance(value, (int, float, str, bool)):
            return ConstantVariable(value)
        elif isinstance(value, (paddle.Tensor, ProxyTensor)):
            return TensorVariable(value, graph)
        return
        raise RuntimeError(
            f"Don't Implement a value binding method for type: `{type(value)}`"
        )


class ConstantVariable(VariableTracker):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __mul__(self, other):
        if not isinstance(other, ConstantVariable):
            return NotImplemented
        var = VariableTrackerFactory.from_value(
            self.value * other.value, self.graph
        )
        return var

    def __add__(self, other):
        if not isinstance(other, ConstantVariable):
            return NotImplemented
        var = VariableTrackerFactory.from_value(
            self.value + other.value, self.graph
        )
        return var


class TensorVariable(VariableTracker):
    def __init__(self, tensor, graph):
        super().__init__()
        self.leaf = False
        if isinstance(tensor, (paddle.Tensor)):
            self.value = ProxyTensorContext().from_tensor(tensor)
            self.leaf = True
        elif isinstance(tensor, ProxyTensor):
            self.value = tensor
        self.graph = graph

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
