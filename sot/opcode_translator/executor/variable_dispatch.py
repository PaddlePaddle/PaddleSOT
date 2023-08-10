from __future__ import annotations

import math
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
from .dispatch_functions import (
    operator_in,
    operator_not_in,
    raise_break_graph_fn,
    tensor_numel,
)
from .dispatcher import Dispatcher, optional
from .tracker import ConstTracker, DummyTracker
from .variables import (
    ConstantVariable,
    ContainerVariable,
    EnumerateVariable,
    VariableBase,
    VariableFactory,
)

if TYPE_CHECKING:
    from .variables import DataVariable, NumpyVariable, TensorVariable


def raise_err_handle(error):
    def inner(*args, **kwargs):
        raise error

    return inner


# in
Dispatcher.register(
    operator_in,
    ("VariableBase", "IterVariable"),
    raise_err_handle(BreakGraphError("Codes like: `variable in iterator`.")),
)

Dispatcher.register(
    operator_in,
    ("TensorVariable", "VariableBase"),
    lambda left, right: VariableFactory.from_value(
        left.id
        in [
            x.id
            for x in right.get_py_value(allow_tensor=True)
            if hasattr(x, "id")
        ],
        left.graph,
        tracker=DummyTracker([left, right]),
    ),
)

Dispatcher.register(
    operator_in,
    ("VariableBase", "VariableBase"),
    lambda left, right: VariableFactory.from_value(
        left.get_py_value(allow_tensor=True)
        in right.get_py_value(allow_tensor=True),
        left.graph,
        tracker=DummyTracker([left, right]),
    ),
)

Dispatcher.register(
    operator_not_in,
    ("VariableBase", "IterVariable"),
    raise_err_handle(
        BreakGraphError("Codes like: `variable not in iterator`.")
    ),
)

Dispatcher.register(
    operator_not_in,
    ("TensorVariable", "VariableBase"),
    lambda left, right: VariableFactory.from_value(
        left.id
        not in [
            x.id
            for x in right.get_py_value(allow_tensor=True)
            if hasattr(x, "id")
        ],
        left.graph,
        tracker=DummyTracker([left, right]),
    ),
)

Dispatcher.register(
    operator_not_in,
    ("VariableBase", "VariableBase"),
    lambda left, right: VariableFactory.from_value(
        left.get_py_value(allow_tensor=True)
        not in right.get_py_value(allow_tensor=True),
        left.graph,
        tracker=DummyTracker([left, right]),
    ),
)

# dict
Dispatcher.register(
    dict.get,
    ("DictVariable", "ConstantVariable", optional("VariableBase")),
    lambda var, key, default=None: var.get(key.get_py_value(), default),
)
Dispatcher.register(
    dict.keys,
    ("DictVariable",),
    lambda var: var.keys(),
)

Dispatcher.register(
    operator.not_,
    ("VariableBase",),
    lambda x: VariableFactory.from_value(
        not x.get_py_value(allow_tensor=False), x.graph, DummyTracker([x])
    ),
)

Dispatcher.register(
    dict.values,
    ("DictVariable",),
    lambda var: var.values(),
)
Dispatcher.register(
    dict.items,
    ("DictVariable",),
    lambda var: var.items(),
)
Dispatcher.register(
    dict.setdefault,
    ("DictVariable", "ConstantVariable", optional("VariableBase")),
    lambda var, key, default=None: var.setdefault(key.get_py_value(), default),
)
Dispatcher.register(
    dict.update,
    ("DictVariable", "DictVariable"),
    lambda var, other: var.update(other),
)
Dispatcher.register(
    dict.copy,
    ("DictVariable",),
    lambda var: var.copy(),
)
Dispatcher.register(
    dict.clear,
    ("DictVariable",),
    lambda var: var.clear(),
)
Dispatcher.register(
    dict.pop,
    ("DictVariable", "ConstantVariable"),
    lambda var, key: var.pop(key.get_py_value()),
)
Dispatcher.register(
    dict.pop,
    ("DictVariable", "ConstantVariable", "VariableBase"),
    lambda var, key, default: var.pop(key.get_py_value(), default),
)
Dispatcher.register(
    dict.popitem,
    ("DictVariable",),
    lambda var: var.popitem(),
)

# tuple
Dispatcher.register(
    tuple,
    ("ContainerVariable | EnumerateVariable",),
    lambda var: VariableFactory.from_value(
        tuple(var.get_wrapped_items()),
        graph=var.graph,
        tracker=DummyTracker([var]),
    ),
)
Dispatcher.register(
    tuple,
    ("SequenceIterVariable",),
    lambda var: VariableFactory.from_value(
        tuple(var.to_list()),
        graph=var.graph,
        tracker=DummyTracker([var]),
    ),
)
Dispatcher.register(
    tuple.count,
    ("TupleVariable", "VariableBase"),
    lambda var, value: var.count(value),
)
Dispatcher.register(
    tuple.index,
    ("TupleVariable", "VariableBase"),
    lambda var, value: var.index(value),
)

# list
Dispatcher.register(
    list,
    ("ContainerVariable | EnumerateVariable",),
    lambda var: VariableFactory.from_value(
        list(var.get_wrapped_items()),
        graph=var.graph,
        tracker=DummyTracker([var]),
    ),
)

Dispatcher.register(
    list,
    ("IterVariable",),
    lambda var: VariableFactory.from_value(
        var.to_list(),
        graph=var.graph,
        tracker=DummyTracker([var]),
    ),
)
Dispatcher.register(
    list.extend,
    ("ListVariable", "ListVariable | TupleVariable"),
    lambda var, other: var.extend(other),
)
Dispatcher.register(
    list.append,
    ("ListVariable", "VariableBase"),
    lambda var, other: var.append(other),
)
Dispatcher.register(
    list.insert,
    ("ListVariable", "ConstantVariable", "VariableBase"),
    lambda var, index, obj: var.insert(index.get_py_value(), obj),
)
Dispatcher.register(
    list.remove,
    ("ListVariable", "VariableBase"),
    lambda var, other: var.remove(other),
)
Dispatcher.register(
    list.pop,
    ("ListVariable", optional("ConstantVariable")),
    lambda var, index=None: var.pop(index),
)
Dispatcher.register(
    list.clear,
    ("ListVariable",),
    lambda var: var.clear(),
)
Dispatcher.register(
    list.sort,
    ("ListVariable",),
    lambda var: var.sort(),
)
Dispatcher.register(
    list.reverse,
    ("ListVariable",),
    lambda var: var.reverse(),
)
Dispatcher.register(
    list.copy,
    ("ListVariable",),
    lambda var: var.copy(),
)
Dispatcher.register(
    list.count,
    ("ListVariable", "VariableBase"),
    lambda var, obj: var.count(obj),
)
Dispatcher.register(
    list.index,
    ("ListVariable", "VariableBase"),
    lambda var, obj: var.index(obj),
)
Dispatcher.register(
    operator.add,
    ("ListVariable", "ListVariable"),
    lambda var, other: var.concat(other),
)
Dispatcher.register(
    operator.add,
    ("TupleVariable", "TupleVariable"),
    lambda var, other: var.concat(other),
)
Dispatcher.register(
    operator.mul,
    ("ListVariable | TupleVariable", "ConstantVariable"),
    lambda var, other: var.repeat(other),
)

# getattr
Dispatcher.register(
    getattr,
    ("VariableBase", "ConstantVariable", optional("VariableBase")),
    lambda var, name, default=None: (
        var.graph.add_global_guarded_variable(name),
        var.getattr(name.get_py_value(), default),
    )[1],
)
# len
Dispatcher.register(
    len,
    ("ContainerVariable | PaddleLayerVariable",),
    lambda var: var.len(),
)


# range
# stop
Dispatcher.register(
    range,
    ("ConstantVariable",),
    lambda stop: VariableFactory.from_value(
        range(stop.get_py_value()),
        graph=stop.graph,
        tracker=DummyTracker([stop]),
    ),
)

# start, stop
Dispatcher.register(
    range,
    ("ConstantVariable", "ConstantVariable"),
    lambda start, stop: VariableFactory.from_value(
        range(start.get_py_value(), stop.get_py_value()),
        graph=stop.graph,
        tracker=DummyTracker([start, stop]),
    ),
)
# start, stop, step
Dispatcher.register(
    range,
    ("ConstantVariable", "ConstantVariable", "ConstantVariable"),
    lambda start, stop, step: VariableFactory.from_value(
        range(start.get_py_value(), stop.get_py_value(), step.get_py_value()),
        graph=stop.graph,
        tracker=DummyTracker([start, stop, step]),
    ),
)
# TODO(zmh): Modify
# enumerate
Dispatcher.register(
    enumerate,
    (
        "ListVariable | TupleVariable | RangeVariable | DictVariable | TensorVariable | PaddleLayerVariable",
    ),
    lambda var: EnumerateVariable.from_iterator(
        var, graph=var.graph, tracker=DummyTracker([var])
    ),
)


# reversed
@Dispatcher.register_decorator(reversed)
def dispatch_reversed(var: ContainerVariable):
    from .tracker import DanglingTracker
    from .variables import BuiltinVariable, SequenceIterVariable

    length_var = BuiltinVariable(len, var.graph, DanglingTracker())(var)
    assert isinstance(length_var, ConstantVariable)
    getitem = BuiltinVariable(operator.getitem, var.graph, DanglingTracker())
    out = reversed([getitem(var, i) for i in range(length_var.get_py_value())])
    out_var = VariableFactory.from_value(
        list(out), graph=var.graph, tracker=DummyTracker([var])
    )
    return SequenceIterVariable(
        out_var,
        graph=var.graph,
        tracker=DummyTracker([var]),
    )


# isinstance
Dispatcher.register(
    isinstance,
    ("TensorVariable", "VariableBase"),
    lambda left, right: ConstantVariable.wrap_literal(
        isinstance(
            paddle.to_tensor(0),
            right.get_py_value(allow_tensor=True),
        ),
        left.graph,
    ),
)

Dispatcher.register(
    isinstance,
    ("VariableBase", "VariableBase"),
    lambda left, right: ConstantVariable.wrap_literal(
        isinstance(
            left.get_py_value(allow_tensor=True),
            right.get_py_value(allow_tensor=True),
        ),
        left.graph,
    ),
)

# bool
Dispatcher.register(
    bool,
    ("ContainerVariable",),
    lambda var: var.bool(),
)
Dispatcher.register(
    bool,
    ("ConstantVariable",),
    lambda var: var.bool(),
)
Dispatcher.register(
    operator.truth,
    ("ContainerVariable",),
    lambda var: var.bool(),
)
Dispatcher.register(
    operator.truth,
    ("ConstantVariable",),
    lambda var: var.bool(),
)

# str
Dispatcher.register(
    str,
    ("ConstantVariable",),
    lambda var: var.str(),
)


@Dispatcher.register_decorator(str.format)
def str_format(var: ConstantVariable, *args: ConstantVariable):
    return var.format(*args)


Dispatcher.register(
    str.lower,
    ("ConstantVariable",),
    lambda var: var.lower(),
)

# getitem
# TODO: Should pass its Variable into the getitem and perform operations such as getting value in the getitem. like this:https://github.com/PaddlePaddle/PaddleSOT/pull/198#discussion_r1241110949
Dispatcher.register(
    operator.getitem,
    (
        "TensorVariable",
        "Any",
    ),
    lambda var, key: var.getitem(
        VariableFactory.from_value(
            key, graph=var.graph, tracker=ConstTracker(key)
        )
    ),
)

Dispatcher.register(
    operator.getitem,
    (
        "VariableBase",
        "int | str",
    ),
    lambda var, key: var.getitem(
        VariableFactory.from_value(
            key, graph=var.graph, tracker=ConstTracker(key)
        )
    ),
)

Dispatcher.register(
    operator.getitem,
    (
        "VariableBase",
        "ConstantVariable | SliceVariable",
    ),
    lambda var, key: var.getitem(key),
)

# setitem
Dispatcher.register(
    operator.setitem,
    (
        "VariableBase",
        "int | str | ConstantVariable | TensorVariable",
        "int | str | ConstantVariable | TensorVariable",
    ),
    lambda var, key, value: var.setitem(key.get_py_value(), value),
)

# delitem
Dispatcher.register(
    operator.delitem,
    (
        "VariableBase",
        "int | str | TensorVariable",
    ),
    lambda var, key: var.delitem(key),
)
Dispatcher.register(
    operator.delitem,
    (
        "VariableBase",
        "ConstantVariable",
    ),
    lambda var, key: var.delitem(key.get_py_value()),
)


# TensorVariable
Dispatcher.register(
    paddle.is_tensor,
    ("TensorVariable",),
    lambda var: var.is_tensor(),
)
Dispatcher.register(
    paddle.is_complex,
    ("TensorVariable",),
    lambda var: var.is_complex(),
)
Dispatcher.register(
    paddle.is_integer,
    ("TensorVariable",),
    lambda var: var.is_integer(),
)
Dispatcher.register(
    paddle.is_floating_point,
    ("TensorVariable",),
    lambda var: var.is_floating_point(),
)
Dispatcher.register(
    paddle.rank,
    ("TensorVariable",),
    lambda var: var.ndim,
)

Dispatcher.register(
    operator.is_,
    ("TensorVariable", "TensorVariable"),
    lambda var, other: VariableFactory.from_value(
        var.get_symbol() == other.get_symbol(),
        var.graph,
        tracker=DummyTracker([var, other]),
    ),
)

Dispatcher.register(
    operator.is_,
    ("TensorVariable", "VariableBase"),
    lambda var, other: VariableFactory.from_value(
        False,
        var.graph,
        tracker=DummyTracker([var, other]),
    ),
)

Dispatcher.register(
    operator.is_,
    ("VariableBase", "TensorVariable"),
    lambda var, other: VariableFactory.from_value(
        False,
        var.graph,
        tracker=DummyTracker([var, other]),
    ),
)

# VariableBase
Dispatcher.register(
    operator.is_,
    ("VariableBase", "VariableBase"),
    lambda var, other: VariableFactory.from_value(
        var.get_py_value() is other.get_py_value(),
        var.graph,
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
            partial(
                lambda fn, var: VariableFactory.from_value(
                    fn(var.get_py_value()),
                    var.graph,
                    tracker=DummyTracker([var]),
                ),
                unary_fn,
            ),
        )
for binary_fn in BINARY_OPS:
    for magic_method in magic_method_builtin_dispatch(binary_fn):
        Dispatcher.register(
            binary_fn,
            ("ConstantVariable", "ConstantVariable"),
            partial(
                lambda fn, var, other: VariableFactory.from_value(
                    fn(var.get_py_value(), other.get_py_value()),
                    var.graph,
                    tracker=DummyTracker([var, other]),
                ),
                binary_fn,
            ),
        )
# Tensor
fallback_tensor_unary_method = {
    int,
    bool,
    operator.truth,
}

Dispatcher.register(tensor_numel, ("TensorVariable",), lambda x: x.numel())

for unary_fn in UNARY_OPS:
    if unary_fn in fallback_tensor_unary_method:
        Dispatcher.register(
            unary_fn,
            ("TensorVariable",),
            raise_break_graph_fn,
        )
        continue

    if unary_fn is len:
        Dispatcher.register(
            unary_fn,
            ("TensorVariable",),
            lambda x: x.len(),
        )
        continue

    for magic_method in magic_method_builtin_dispatch(unary_fn):
        Dispatcher.register(
            unary_fn,
            ("TensorVariable",),
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
                    if var.get_py_type() is str:
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
        operator(var.get_py_value(), other.get_py_value()),
        var.graph,
        DummyTracker([var, other]),
    )


for binary_fn in BINARY_OPS:
    for magic_method in magic_method_builtin_dispatch(binary_fn):
        Dispatcher.register(
            binary_fn,
            ("DataVariable", "Any"),
            partial(data_variable_binary_dispatcher, operator=binary_fn),
        )
        Dispatcher.register(
            binary_fn,
            ("Any", "DataVariable"),
            partial(data_variable_binary_dispatcher, operator=binary_fn),
        )

for unary_fn in UNARY_OPS:
    for magic_method in magic_method_builtin_dispatch(unary_fn):

        def data_variable_unary_dispatcher(var: DataVariable, fn):
            return VariableFactory.from_value(
                fn(var.get_py_value()),
                var.graph,
                DummyTracker([var]),
            )

        Dispatcher.register(
            unary_fn,
            ("DataVariable",),
            partial(data_variable_unary_dispatcher, fn=unary_fn),
        )


Dispatcher.register(
    math.ceil,
    ("ConstantVariable",),
    lambda var: VariableFactory.from_value(
        math.ceil(var.get_py_value()),
        var.graph,
        tracker=DummyTracker([var]),
    ),
)

Dispatcher.register(
    math.floor,
    ("ConstantVariable",),
    lambda var: VariableFactory.from_value(
        math.floor(var.get_py_value()),
        var.graph,
        tracker=DummyTracker([var]),
    ),
)
