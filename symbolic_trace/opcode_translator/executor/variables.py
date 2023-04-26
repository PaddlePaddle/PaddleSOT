import paddle

from ...proxy_tensor import ProxyTensor, ProxyTensorContext
from ...utils import NameGenerator
from ...utils.exceptions import InnerError


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
        if not self.source:
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
            assert graph is not None
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
        var = VariableTrackerFactory.from_value(self.value * other.value, None)
        return var

    def __add__(self, other):
        if not isinstance(other, ConstantVariable):
            return NotImplemented
        var = VariableTrackerFactory.from_value(self.value + other.value, None)
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


class ListVariable(VariableTracker):
    def __init__(self, val_list):
        super().__init__()
        # everything in stack is VariableTracker, so just accept the input list is ok
        self._list = val_list

    def __len__(self):
        return len(self._list)

    def __getitem__(self, key):
        try:
            assert isinstance(key, ConstantVariable)
            index = int(key.value)
        except:
            raise InnerError(
                "[ListVariable]: recieved {key}:{key.value} as key."
            )

        return self._list[index]

    def __setitem__(self, key, value):
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


class TupleVariable(VariableTracker):
    def __init__(self, val_tuple):
        super().__init__()
        self._tuple = val_tuple

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
