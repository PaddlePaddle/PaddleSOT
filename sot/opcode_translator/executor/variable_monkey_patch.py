from functools import partial

from ...utils.exceptions import NotImplementException
from ...utils.monkey_patch import (
    binary_operator_methods,
    do_monkey_patch,
    unary_operator_methods,
)
from .dispatcher import Dispatcher, MagicMethodDispatcher
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

# NOTE(SigureMo): Don't directly capture free var inside for-loop, use partial instead.
# ```python
# lambdas = []
# for i in range(10):
#     lambdas.append(lambda: i)
# for fn in lambdas:
#     print(fn()) # result is 9, 9, 9, 9, 9, 9, 9, 9, 9, 9
# ```
# Rewrite by partial:
# ```python
# lambdas = []
# for i in range(10):
#     lambdas.append(partial(lambda i: i, i))
# for fn in lambdas:
#     print(fn()) # result is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
# ```

# Constant
for binary_fn, (
    magic_name,
    reverse_magic_name,
) in MagicMethodDispatcher.binary_op_names.items():
    Dispatcher.register(
        binary_fn,
        ("ConstantVariable", "ConstantVariable"),
        {},
        partial(
            lambda fn, var, other: VariableFactory.from_value(
                fn(var.get_value(), other.get_value()),
                None,
                tracker=DummyTracker([var, other]),
            ),
            binary_fn,
        ),
    )
for unary_fn, magic_name in MagicMethodDispatcher.unary_op_names.items():
    Dispatcher.register(
        unary_fn,
        ("ConstantVariable",),
        {},
        partial(
            lambda fn, var: VariableFactory.from_value(
                fn(var.get_value()), None, tracker=DummyTracker([var])
            ),
            unary_fn,
        ),
    )
# Tensor
for binary_fn, (
    magic_name,
    reverse_magic_name,
) in MagicMethodDispatcher.binary_op_names.items():
    # TODO: skip __mod__ for str and TensorVariable
    Dispatcher.register(
        binary_fn,
        (
            "TensorVariable",
            "TensorVariable | ConstantVariable",
        ),
        {},
        partial(
            lambda magic_name, var, other: var.graph.call_tensor_method(
                magic_name, var, other
            ),
            magic_name,
        ),
    )
    Dispatcher.register(
        binary_fn,
        (
            "ConstantVariable",
            "TensorVariable",
        ),
        {},
        partial(
            lambda reverse_magic_name, var, other: other.graph.call_tensor_method(
                reverse_magic_name, other, var
            ),
            reverse_magic_name,
        ),
    )
for unary_fn, magic_name in MagicMethodDispatcher.unary_op_names.items():
    Dispatcher.register(
        unary_fn,
        ("TensorVariable",),
        {},
        lambda var: var.graph.call_tensor_method(magic_name, var),
    )
