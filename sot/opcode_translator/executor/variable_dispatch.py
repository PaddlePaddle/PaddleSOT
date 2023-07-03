from __future__ import annotations

import operator
from functools import partial
from typing import TYPE_CHECKING

import paddle

from ...utils import BreakGraphError, NotImplementException
from ...utils.magic_methods import (
    BINARY_OPS,
    UNARY_OPS,
    magic_method_builtin_dispatch,
)
from .dispatcher import Dispatcher
from .tracker import DummyTracker
from .variables import VariableBase, VariableFactory

if TYPE_CHECKING:
    from .variables import (
        ConstantVariable,
        DataVariable,
        NumpyVariable,
        TensorVariable,
    )


# dict
Dispatcher.register(
    dict.keys,
    ("DictVariable",),
    {},
    lambda var: var.keys(),
)
Dispatcher.register(
    dict.values,
    ("DictVariable",),
    {},
    lambda var: var.values(),
)
Dispatcher.register(
    dict.items,
    ("DictVariable",),
    {},
    lambda var: var.items(),
)
Dispatcher.register(
    dict.update,
    ("DictVariable", "DictVariable"),
    {},
    lambda var, other: var.update(other),
)
# list
Dispatcher.register(
    list.extend,
    ("ListVariable", "ListVariable | TupleVariable"),
    {},
    lambda var, other: var.extend(other),
)
Dispatcher.register(
    list.append,
    ("ListVariable", "VariableBase"),
    {},
    lambda var, other: var.append(other),
)
Dispatcher.register(
    list.insert,
    ("ListVariable", "ConstantVariable", "VariableBase"),
    {},
    lambda var, index, obj: var.insert(index, obj),
)
Dispatcher.register(
    list.remove,
    ("ListVariable", "VariableBase"),
    {},
    lambda var, other: var.remove(other),
)
Dispatcher.register(
    list.pop,
    ("ListVariable", "ConstantVariable"),
    {},
    lambda var, other: var.pop(other),
)
Dispatcher.register(
    list.pop,
    ("ListVariable",),
    {},
    lambda var: var.pop(),
)
Dispatcher.register(
    list.clear,
    ("ListVariable",),
    {},
    lambda var: var.clear(),
)
Dispatcher.register(
    list.sort,
    ("ListVariable",),
    {},
    lambda var: var.sort(),
)
Dispatcher.register(
    list.reverse,
    ("ListVariable",),
    {},
    lambda var: var.reverse(),
)
Dispatcher.register(
    operator.add,
    ("ListVariable", "ListVariable"),
    {},
    lambda var, other: var.concat(other),
)
Dispatcher.register(
    operator.add,
    ("TupleVariable", "TupleVariable"),
    {},
    lambda var, other: var.concat(other),
)
Dispatcher.register(
    operator.mul,
    ("ListVariable | TupleVariable", "ConstantVariable"),
    {},
    lambda var, other: var.repeat(other),
)
# getattr
# TODO(SigureMo): Unify these to a single function
# TODO(SigureMo): Default argument will case duplicated code.
Dispatcher.register(
    getattr,
    ("VariableBase", "str"),
    {},
    lambda var, name: var.getattr(name),
)
Dispatcher.register(
    getattr,
    ("VariableBase", "str", "VariableBase"),
    {},
    lambda var, name: var.getattr(name),
)
Dispatcher.register(
    getattr,
    ("VariableBase", "ConstantVariable"),
    {},
    lambda var, name, default: var.getattr(name.get_value(), default),
)
Dispatcher.register(
    getattr,
    ("VariableBase", "ConstantVariable", "VariableBase"),
    {},
    lambda var, name, default: var.getattr(name.get_value(), default),
)
# len
Dispatcher.register(
    len,
    ("ContainerVariable",),
    {},
    lambda var: var.len(),
)
# bool
Dispatcher.register(
    bool,
    ("ContainerVariable",),
    {},
    lambda var: var.bool(),
)
Dispatcher.register(
    bool,
    ("ConstantVariable",),
    {},
    lambda var: var.bool(),
)
Dispatcher.register(
    operator.truth,
    ("ContainerVariable",),
    {},
    lambda var: var.bool(),
)
Dispatcher.register(
    operator.truth,
    ("ConstantVariable",),
    {},
    lambda var: var.bool(),
)

# getitem
# TODO: Should pass its Variable into the getitem and perform operations such as getting value in the getitem. like this:https://github.com/PaddlePaddle/PaddleSOT/pull/198#discussion_r1241110949
Dispatcher.register(
    operator.getitem,
    (
        "TensorVariable",
        "Any",
    ),
    {},
    lambda var, key: var.getitem(key),
)

Dispatcher.register(
    operator.getitem,
    (
        "VariableBase",
        "int | str | TensorVariable | slice",
    ),
    {},
    lambda var, key: var.getitem(key),
)
Dispatcher.register(
    operator.getitem,
    (
        "VariableBase",
        "ConstantVariable | SliceVariable",
    ),
    {},
    lambda var, key: var.getitem(key.get_value()),
)

# setitem
Dispatcher.register(
    operator.setitem,
    (
        "VariableBase",
        "int | str | ConstantVariable | TensorVariable",
        "int | str | ConstantVariable | TensorVariable",
    ),
    {},
    lambda var, key, value: var.setitem(key.get_value(), value),
)

# delitem
Dispatcher.register(
    operator.delitem,
    (
        "VariableBase",
        "int | str | TensorVariable",
    ),
    {},
    lambda var, key: var.delitem(key),
)
Dispatcher.register(
    operator.delitem,
    (
        "VariableBase",
        "ConstantVariable",
    ),
    {},
    lambda var, key: var.delitem(key.get_value()),
)


# TensorVariable
Dispatcher.register(
    paddle.is_tensor,
    ("TensorVariable",),
    {},
    lambda var: var.is_tensor(),
)
Dispatcher.register(
    paddle.is_complex,
    ("TensorVariable",),
    {},
    lambda var: var.is_complex(),
)
Dispatcher.register(
    paddle.is_integer,
    ("TensorVariable",),
    {},
    lambda var: var.is_integer(),
)
Dispatcher.register(
    paddle.is_floating_point,
    ("TensorVariable",),
    {},
    lambda var: var.is_floating_point(),
)
Dispatcher.register(
    paddle.rank,
    ("TensorVariable",),
    {},
    lambda var: var.ndim,
)

Dispatcher.register(
    operator.is_,
    ("TensorVariable", "TensorVariable"),
    {},
    lambda var, other: VariableFactory.from_value(
        var.get_symbol() == other.get_symbol(),
        None,
        tracker=DummyTracker([var, other]),
    ),
)

Dispatcher.register(
    operator.is_,
    ("TensorVariable", "VariableBase"),
    {},
    lambda var, other: VariableFactory.from_value(
        False,
        None,
        tracker=DummyTracker([var, other]),
    ),
)

Dispatcher.register(
    operator.is_,
    ("VariableBase", "TensorVariable"),
    {},
    lambda var, other: VariableFactory.from_value(
        False,
        None,
        tracker=DummyTracker([var, other]),
    ),
)

# VariableBase
Dispatcher.register(
    operator.is_,
    ("VariableBase", "VariableBase"),
    {},
    lambda var, other: VariableFactory.from_value(
        var.get_value() is other.get_value(),
        None,
        tracker=DummyTracker([var, other]),
    ),
)


@Dispatcher.register_decorator(operator.is_not)
def is_not_func(var: VariableBase, other: VariableBase):
    handler = Dispatcher.dispatch(operator.is_, var, other)
    if handler is None:
        raise NotImplementException(
            f"Not found implementation operator.is for {var} and {other}."
        )
    return handler(var, other).bool_not()


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
for unary_fn in UNARY_OPS:
    for magic_method in magic_method_builtin_dispatch(unary_fn):
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
for binary_fn in BINARY_OPS:
    for magic_method in magic_method_builtin_dispatch(binary_fn):
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
# Tensor
for unary_fn in UNARY_OPS:
    # Tensor doesn't support unary +, skip it
    if unary_fn in {operator.pos, len}:
        continue
    for magic_method in magic_method_builtin_dispatch(unary_fn):
        Dispatcher.register(
            unary_fn,
            ("TensorVariable",),
            {},
            partial(
                lambda magic_name, var: var.graph.call_tensor_method(
                    magic_name, var
                ),
                magic_method.name,
            ),
        )
for binary_fn in BINARY_OPS:
    for magic_method in magic_method_builtin_dispatch(binary_fn):
        # skip all inplace magic method name, we will dispatch it to non-inplace
        # magic methods
        if magic_method.is_inplace:
            continue

        if not magic_method.is_reverse:
            Dispatcher.register(
                binary_fn,
                (
                    "TensorVariable",
                    "TensorVariable | ConstantVariable | NumpyVariable",
                ),
                {},
                partial(
                    lambda magic_name, var, other: var.graph.call_tensor_method(
                        magic_name, var, other
                    ),
                    magic_method.name,
                ),
            )
        else:
            # skip __mod__ for str and TensorVariable
            if magic_method.name == "__rmod__":

                @Dispatcher.register_decorator(operator.mod)
                def tensor_mod_dispatcher(
                    var: ConstantVariable, other: TensorVariable
                ):
                    if isinstance(var.get_value(), str):
                        raise BreakGraphError(
                            "(ConstantVariable % TensorVariable) raise a callback. "
                        )
                    raise NotImplementException(
                        "Tensor doesn't support __rmod__"
                    )

            else:
                Dispatcher.register(
                    binary_fn,
                    (
                        "ConstantVariable | NumpyVariable",
                        "TensorVariable",
                    ),
                    {},
                    partial(
                        lambda reverse_magic_name, var, other: other.graph.call_tensor_method(
                            reverse_magic_name, other, var
                        ),
                        magic_method.name,
                    ),
                )
# Register dispatch for NumpyVariable: fallback !
for unary_fn in UNARY_OPS:
    for magic_method in magic_method_builtin_dispatch(unary_fn):

        @Dispatcher.register_decorator(unary_fn)
        def numpy_unary_dispatcher(var: NumpyVariable):
            raise NotImplementException(
                'Numpy operator need fallback to dygraph'
            )


for binary_fn in BINARY_OPS:
    for magic_method in magic_method_builtin_dispatch(binary_fn):

        @Dispatcher.register_decorator(binary_fn)
        def numpy_binary_dispatcher(var: NumpyVariable, other: NumpyVariable):
            raise NotImplementException(
                'Numpy operator need fallback to dygraph'
            )


# Register dispatch for DataVariable: directy call and return a wrapped variable.
def data_variable_binary_dispatcher(var, other, operator):
    return VariableFactory.from_value(
        operator(var.get_value(), other.get_value()),
        var.graph,
        DummyTracker([var, other]),
    )


for binary_fn in BINARY_OPS:
    for magic_method in magic_method_builtin_dispatch(binary_fn):
        Dispatcher.register(
            binary_fn,
            ("DataVariable", "Any"),
            {},
            partial(data_variable_binary_dispatcher, operator=binary_fn),
        )
        Dispatcher.register(
            binary_fn,
            ("Any", "DataVariable"),
            {},
            partial(data_variable_binary_dispatcher, operator=binary_fn),
        )

for unary_fn in UNARY_OPS:
    for magic_method in magic_method_builtin_dispatch(unary_fn):

        def data_variable_unary_dispatcher(var: DataVariable, fn):
            return VariableFactory.from_value(
                fn(var.get_value()),
                var.graph,
                DummyTracker([var]),
            )

        Dispatcher.register(
            unary_fn,
            ("DataVariable",),
            {},
            partial(data_variable_unary_dispatcher, fn=unary_fn),
        )
