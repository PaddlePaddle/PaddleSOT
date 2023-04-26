import paddle

from ...proxy_tensor import ProxyTensor, ProxyTensorContext
from ...utils import NameGenerator
from ...utils.exceptions import InnerError
from .source import GetItemSource


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
        elif isinstance(value, list):
            return ListVariable(value)
        elif isinstance(value, tuple):
            return TupleVariable(value)
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

        retval = self._list[index]

        # if list is an input of funciton, we need make sure __getitem__ returns a VariableTracker
        if not isinstance(retval, VariableTracker):
            retval = VariableTrackerFactory.from_value(retval)
            if self.source is not None:
                # the retval is from self at place index
                retval.set_source(GetItemSource(self, key))
                # set it back, it is a bit ugly
                self._list[index] = retval

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
        self._tuple = val_tuple  # exactly it is a list
        # (need replace item with VaraibleTracker)

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

        retval = self._tuple[index]
        if not isinstance(retval, VariableTracker):
            retval = VariableTrackerFactory.from_value(retval)
            if self.source is not None:
                retval.set_source(GetItemSource(self, key))
                self._tuple[index] = retval
            self._tuple[index] = retval

        return retval

    def __setitem__(self, key, value):
        raise InnerError("[TupleVariable]: setitem is not allowed.")

    def __delitem__(self, key):
        raise InnerError("[TupleVariable]: delitem is not allowed.")
