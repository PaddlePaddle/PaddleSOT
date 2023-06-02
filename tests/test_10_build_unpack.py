# BUILD_TUPLE_UNPACK (new)
# BUILD_LIST_UNPACK (new)
# BUILD_TUPLE_UNPACK_WITH_CALL (new)
# CALL_FUNCTION_EX (new)
# BUILD_MAP_UNPACK (new)
# LIST_EXTEND (new)
# LIST_TO_TUPLE (new)
# DICT_UPDATE (new)
# DICT_MERGE (new)

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def build_tuple_unpack(x: tuple[paddle.Tensor], y: tuple[paddle.Tensor]):
    z = (*x, *y)

    return z[0] + 1


def build_list_unpack(x: list[paddle.Tensor], y: list[paddle.Tensor]):
    z = [*x, *y]
    return z[0] + 1


def build_tuple_unpack_with_call(
    x: tuple[paddle.Tensor], y: tuple[paddle.Tensor]
):
    z = build_tuple_unpack_with_call_inner(*x, *y)
    return z[0] + 1


def build_tuple_unpack_with_call_inner(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor
):
    z = (a, b, c, d)
    return z


def build_map_unpack(x: dict[str, paddle.Tensor], y: dict[str, paddle.Tensor]):
    z = {**x, **y}
    return z["a"] + 1


def build_map_unpack_with_call_inner(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor
):
    z = {"a": a, "b": b, "c": c, "d": d}
    return z


def build_map_unpack_with_call(
    x: dict[str, paddle.Tensor], y: dict[str, paddle.Tensor]
):
    z = build_map_unpack_with_call_inner(**x, **y)
    return z["a"] + 1


class TestExecutor(TestCaseBase):
    def test_simple_list(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        c = paddle.to_tensor(3)
        d = paddle.to_tensor(4)

        self.assert_results(build_tuple_unpack, (a, b), (c, d))
        self.assert_results(build_list_unpack, [a, b], [c, d])
        self.assert_results(build_tuple_unpack_with_call, (a, b), (c, d))
        self.assert_results(
            build_map_unpack, {"a": a, "b": b}, {"c": c, "d": d}
        )
        self.assert_results(
            build_map_unpack_with_call, {"a": a, "b": b}, {"c": c, "d": d}
        )


if __name__ == "__main__":
    # t = TestExecutor()
    # t.test_simple()
    unittest.main()
