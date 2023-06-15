from __future__ import annotations

import inspect
import operator
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from ...utils import InnerError

if TYPE_CHECKING:
    # We should not depend on variables in this file at runtime.

    T = TypeVar("T")
    Args = tuple[T, ...]
    Kwargs = dict[str, T]


def format_type(type_: type[Any] | tuple[type[Any], ...]) -> str:
    if not isinstance(type_, tuple):
        type_ = (type_,)
    return " | ".join([t.__name__ for t in type_])


def convert_annotation_to_type(type_str: str) -> tuple[type[Any], ...]:
    import builtins

    from . import variables

    type_str = type_str.strip()
    if type_str == "Any":
        type_str = "object"

    if "|" in type_str:
        return reduce(
            operator.add, map(convert_annotation_to_type, type_str.split("|"))
        )

    search_namespaces = [variables, builtins]
    for namespace in search_namespaces:
        if hasattr(namespace, type_str):
            return (getattr(namespace, type_str),)
    raise InnerError(f"Cannot find type {type_str} in {search_namespaces}")


class Pattern:
    type_strings: Args[str]
    kwtype_strings: Kwargs[str]

    def __init__(
        self,
        *types: str,
        **kwtypes: str,
    ):
        self.type_strings = types
        self.kwtype_strings = kwtypes

    @cached_property
    def types(self) -> Args[tuple[type[Any], ...]]:
        return tuple(
            convert_annotation_to_type(type_) for type_ in self.type_strings
        )

    @cached_property
    def kwtypes(self) -> Kwargs[tuple[type[Any], ...]]:
        return {
            name: convert_annotation_to_type(type_)
            for name, type_ in self.kwtype_strings.items()
        }

    def match_inputs(self, *args: Any, **kwargs: Any) -> bool:
        if len(args) != len(self.types):
            return False
        if any(name not in kwargs for name in self.kwtypes.keys()):
            return False
        return all(
            isinstance(arg, type_) for arg, type_ in zip(args, self.types)
        ) and all(
            isinstance(kwargs[name], type_)
            for name, type_ in self.kwtypes.items()
        )

    def __repr__(self) -> str:
        types_repr = ", ".join([format_type(type_) for type_ in self.types])
        kwtypes_repr = ", ".join(
            [
                f"{name}={format_type(type_)}"
                for name, type_ in self.kwtypes.items()
            ]
        )
        return f"Pattern({types_repr}, {kwtypes_repr})"


class Dispatcher:
    handlers: dict[
        Callable[..., Any], list[tuple[Pattern, Callable[..., Any]]]
    ] = {}

    @classmethod
    def register(
        cls,
        fn: Callable[..., Any],
        types: tuple[str, ...],
        kwtypes: dict[str, str],
        handler: Callable[..., Any],
    ):
        if fn not in cls.handlers:
            cls.handlers[fn] = []
        cls.handlers[fn].append((Pattern(*types, **kwtypes), handler))

    @classmethod
    def register_decorator(cls, fn: Callable[..., Any]):
        def decorator(handler: Callable[..., Any]):
            signature = inspect.signature(handler)
            types: list[str] = []
            for name, param in signature.parameters.items():
                if param.annotation == param.empty:
                    types.append("Any")
                elif (
                    param.kind == param.VAR_POSITIONAL
                    or param.kind == param.VAR_KEYWORD
                ):
                    raise InnerError("Not support varargs in decorator mode.")
                else:
                    types.append(str(param.annotation))
            cls.register(fn, tuple(types), {}, handler)
            return None

        return decorator

    @classmethod
    def dispatch(
        cls, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Callable[..., Any] | None:
        if fn not in cls.handlers:
            return None
        for pattern, handler in cls.handlers[fn]:
            if pattern.match_inputs(*args, **kwargs):
                return handler
        return None


class MagicMethodDispatcher:
    binary_op_names: dict[Callable[[Any, Any], Any], tuple[str, str | None]] = {
        operator.add: ("__add__", "__radd__"),
        operator.and_: ("__and__", "__rand__"),
        operator.contains: ("__contains__", None),
        operator.delitem: ("__delitem__", None),
        operator.eq: ("__eq__", "__eq__"),
        operator.floordiv: ("__floordiv__", "__rfloordiv__"),
        operator.ge: ("__ge__", "__le__"),
        operator.getitem: ("__getitem__", None),
        operator.gt: ("__gt__", "__lt__"),
        operator.iadd: ("__iadd__", None),
        operator.iand: ("__iand__", None),
        operator.iconcat: ("__iconcat__", None),
        operator.ifloordiv: ("__ifloordiv__", None),
        operator.ilshift: ("__ilshift__", None),
        operator.imatmul: ("__imatmul__", None),
        operator.imod: ("__imod__", None),
        operator.imul: ("__imul__", None),
        operator.ior: ("__ior__", None),
        operator.ipow: ("__ipow__", None),
        operator.irshift: ("__irshift__", None),
        operator.isub: ("__isub__", None),
        operator.itruediv: ("__itruediv__", None),
        operator.ixor: ("__ixor__", None),
        operator.le: ("__le__", "__ge__"),
        operator.lshift: ("__lshift__", "__rlshift__"),
        operator.lt: ("__lt__", "__gt__"),
        operator.matmul: ("__matmul__", "__rmatmul__"),
        operator.mod: ("__mod__", "__rmod__"),
        operator.mul: ("__mul__", "__rmul__"),
        operator.ne: ("__ne__", "__ne__"),
        operator.or_: ("__or__", "__ror__"),
        operator.pow: ("__pow__", "__rpow__"),
        operator.rshift: ("__rshift__", "__rrshift__"),
        operator.sub: ("__sub__", "__rsub__"),
        operator.truediv: ("__truediv__", "__rtruediv__"),
        operator.xor: ("__xor__", "__rxor__"),
    }
    unary_op_names: dict[Callable[[Any], Any], str] = {
        operator.neg: "__neg__",
        operator.invert: "__invert__",
        operator.pos: "__pos__",
        operator.abs: "__abs__",
        operator.index: "__index__",
        operator.inv: "__inv__",
        operator.invert: "__invert__",
        operator.not_: "__not__",
        operator.pos: "__pos__",
        operator.truth: "__bool__",
        bool: "__bool__",
        abs: "__abs__",
        float: "__float__",
        len: "__len__",
        int: "__int__",
    }
    # TODO(SigureMo): support any, all, sum

    @classmethod
    def dispatch(
        cls, fn: Callable[..., Any], args: Any
    ) -> tuple[Callable[..., Any], bool] | None:
        if fn in cls.binary_op_names:
            assert len(args) == 2, "Binary op should have 2 args."
            left, right = args
            magic_name, reverse_magic_name = cls.binary_op_names[fn]
            if hasattr(left.get_type(), magic_name):
                return getattr(left.get_type(), magic_name), False
            elif reverse_magic_name is not None and hasattr(
                right.get_type(), reverse_magic_name
            ):
                return getattr(right.get_type(), reverse_magic_name), True
        elif fn in cls.unary_op_names:
            assert len(args) == 1, "Unary op should have 1 arg."
            (arg,) = args
            magic_name = cls.unary_op_names[fn]
            if hasattr(arg.get_type().__class__, magic_name):
                return getattr(arg.get_type(), magic_name), False
        return None


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
# getattr
# TODO(SigureMo): Unify these to a single function
Dispatcher.register(
    getattr,
    ("VariableBase", "str"),
    {},
    lambda var, name: var.getattr(name),
)
Dispatcher.register(
    getattr,
    ("VariableBase", "ConstantVariable"),
    {},
    lambda var, name: var.getattr(name.get_value()),
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
    operator.truth,
    ("ContainerVariable",),
    {},
    lambda var: var.bool(),
)
# Tensor
for fn, (
    magic_name,
    reverse_magic_name,
) in MagicMethodDispatcher.binary_op_names.items():
    Dispatcher.register(
        fn,
        (
            "TensorVariable | ConstantVariable",
            "TensorVariable | ConstantVariable",
        ),
        {},
        lambda var, other: var.graph.call_tensor_method(magic_name, var, other),
    )
for fn, magic_name in MagicMethodDispatcher.unary_op_names.items():
    Dispatcher.register(
        fn,
        ("TensorVariable",),
        {},
        lambda var: var.graph.call_tensor_method(magic_name, var),
    )
# # Constant
# for fn, (
#     magic_name,
#     reverse_magic_name,
# ) in MagicMethodDispatcher.binary_op_names.items():
#     Dispatcher.register(
#         fn,
#         ("ConstantVariable", "ConstantVariable"),
#         {},
#         lambda var, other: fn(var.get_value()),
#     )
