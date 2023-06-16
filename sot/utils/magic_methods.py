from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    BinaryOp = Callable[[Any, Any], Any]
    UnaryOp = Callable[[Any], Any]


INPLACE_BINARY_OPS_TO_NON_INPLACE_BINARY_OPS: dict[BinaryOp, BinaryOp] = {
    operator.iadd: operator.add,
    operator.iand: operator.and_,
    operator.iconcat: operator.concat,
    operator.ifloordiv: operator.floordiv,
    operator.ilshift: operator.lshift,
    operator.imatmul: operator.matmul,
    operator.imod: operator.mod,
    operator.imul: operator.mul,
    operator.ior: operator.or_,
    operator.ipow: operator.pow,
    operator.irshift: operator.rshift,
    operator.isub: operator.sub,
    operator.itruediv: operator.truediv,
    operator.ixor: operator.xor,
}

NON_INPLACE_BINARY_OPS_TO_MAGIC_NAMES: dict[
    BinaryOp, tuple[str, str | None]
] = {
    # op fn: (magic name, reverse magic name)
    operator.add: ("__add__", "__radd__"),
    operator.and_: ("__and__", "__rand__"),
    operator.contains: ("__contains__", None),
    operator.delitem: ("__delitem__", None),
    operator.eq: ("__eq__", "__eq__"),
    operator.floordiv: ("__floordiv__", "__rfloordiv__"),
    operator.ge: ("__ge__", "__le__"),
    operator.getitem: ("__getitem__", None),
    operator.gt: ("__gt__", "__lt__"),
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

UNARY_OPS_TO_MAGIC_NAMES: dict[UnaryOp, str] = {
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


INPLACE_BINARY_OPS = set(INPLACE_BINARY_OPS_TO_NON_INPLACE_BINARY_OPS.keys())
NON_INPLACE_BINARY_OPS = set(NON_INPLACE_BINARY_OPS_TO_MAGIC_NAMES.keys())
BINARY_OPS = INPLACE_BINARY_OPS | NON_INPLACE_BINARY_OPS
UNARY_OPS = set(UNARY_OPS_TO_MAGIC_NAMES.keys())


class MagicMethod:
    def __init__(self, name, is_reverse=False):
        self.name = name
        self.is_reverse = is_reverse

    # def apply(self, *args):
    #     if self.is_reverse:
    #         args = args[::-1]
    #     return getattr(args[0], self.name)(*args[1:])
