from ...utils.monkey_patch import (
    binary_operator_methods,
    do_monkey_patch,
    unary_operator_methods,
)
from .variables import ConstantVariable, TensorVariable


# TensorVaraible MonkeyPatch
def tensor_variable_unary_method_builder(method_name):
    def __impl__(self):
        return self.graph.call_tensor_method(method_name, self)

    return __impl__


def tensor_variable_binary_method_builder(method_name):
    def __impl__(self, other):
        if not isinstance(other, (ConstantVariable, TensorVariable)):
            return NotImplemented
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
        return self.apply_unary_operator(method_name)

    return __impl__


def constant_variable_binary_method_builder(method_name):
    def __impl__(self, other):
        return self.apply_binary_operator(other, method_name)

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
