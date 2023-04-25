from __future__ import annotations

import dis

import torch


@torch.compile
def build_tuple_unpack(x: tuple[torch.Tensor], y: tuple[torch.Tensor]):
    z = (*x, *y)
    return z[0] + 1


@torch.compile
def build_list_unpack(x: list[torch.Tensor], y: list[torch.Tensor]):
    z = [*x, *y]
    return z[0] + 1


@torch.compile
def build_tuple_unpack_with_call(
    x: tuple[torch.Tensor], y: tuple[torch.Tensor]
):
    z = build_tuple_unpack_with_call_inner(*x, *y)
    return z[0] + 1


def build_tuple_unpack_with_call_inner(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
):
    z = (a, b, c, d)
    return z


@torch.compile
def build_map_unpack(x: dict[str, torch.Tensor], y: dict[str, torch.Tensor]):
    z = {**x, **y}
    return z["a"] + 1


def build_map_unpack_with_call_inner(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
):
    z = {"a": a, "b": b, "c": c, "d": d}
    return z


@torch.compile
def build_map_unpack_with_call(
    x: dict[str, torch.Tensor], y: dict[str, torch.Tensor]
):
    z = build_map_unpack_with_call_inner(**x, **y)
    return z["a"] + 1


a = torch.as_tensor(1)
b = torch.as_tensor(2)
c = torch.as_tensor(3)
d = torch.as_tensor(4)
build_tuple_unpack((a, b), (c, d))
build_list_unpack([a, b], [c, d])
build_tuple_unpack_with_call((a, b), (c, d))
build_map_unpack({"a": a, "b": b}, {"c": c, "d": d})
build_map_unpack_with_call({"a": a, "b": b}, {"c": c, "d": d})

# Instructions:
# LOAD_FAST
# BUILD_TUPLE_UNPACK (new)
# BUILD_LIST_UNPACK (new)
# BUILD_TUPLE_UNPACK_WITH_CALL (new)
# CALL_FUNCTION_EX (new)
# BUILD_MAP_UNPACK (new)
# STORE_FAST
# BINARY_SUBSCR
# BINARY_ADD
# RETURN_VALUE


# Variables:
# TupleVariable
# ListVariable
# ConstantVariable
# UserFunctionVariable
# TensorVariable
# ConstDictVariable
