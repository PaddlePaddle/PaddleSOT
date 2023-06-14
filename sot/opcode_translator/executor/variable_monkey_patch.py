from sot.utils.exceptions import NotImplementException

from ...utils.monkey_patch import (
    binary_operator_methods,
    do_monkey_patch,
    unary_operator_methods,
)
from .tracker import DummyTracker
from .variables import (
    ConstantVariable,
    NumpyVariable,
    TensorVariable,
    VariableFactory,
)


# TensorVaraible MonkeyPatch
def tensor_variable_unary_method_builder(method_name):
    def __impl__(self):
        return self.graph.call_tensor_method(method_name, self)

    return __impl__


def tensor_variable_binary_method_builder(method_name):
    def __impl__(self, other):
        if not isinstance(other, (ConstantVariable, TensorVariable)):
            raise NotImplemented
        return self.graph.call_tensor_method(method_name, self, other)

    return __impl__


do_monkey_patch(
    TensorVariable, unary_operator_methods, tensor_variable_unary_method_builder
)
do_monkey_patch(
    TensorVariable,
    binary_operator_methods,
    tensor_variable_binary_method_builder,
)


# ConstantVariable MonkeyPatch
def constant_variable_unary_method_builder(method_name):
    def __impl__(self):
        operator = getattr(self.value, method_name)
        var = VariableFactory.from_value(
            operator(),
            None,
            tracker=DummyTracker(
                [
                    self,
                ]
            ),
        )
        return var

    return __impl__


def constant_variable_binary_method_builder(method_name):
    def __impl__(self, other):
        if not isinstance(other, ConstantVariable):
            return NotImplemented
        operator = getattr(self.value, method_name)
        var = VariableFactory.from_value(
            operator(other.value), None, tracker=DummyTracker([self, other])
        )
        return var

    return __impl__


do_monkey_patch(
    ConstantVariable,
    unary_operator_methods,
    constant_variable_unary_method_builder,
)

do_monkey_patch(
    ConstantVariable,
    binary_operator_methods,
    constant_variable_binary_method_builder,
)


# NumpyVariable MonkeyPatch
def numpy_variable_unary_method_builder(method_name):
    def __impl__(self):
        raise NotImplementException('Numpy operator need fallback to dygraph')

    return __impl__


def numpy_variable_binary_method_builder(method_name):
    def __impl__(self, other):
        raise NotImplementException('Numpy operator need fallback to dygraph')

    return __impl__


do_monkey_patch(
    NumpyVariable,
    unary_operator_methods,
    numpy_variable_unary_method_builder,
)

do_monkey_patch(
    NumpyVariable,
    binary_operator_methods,
    numpy_variable_binary_method_builder,
)
