from __future__ import annotations

import unittest
from typing import Iterable

from test_case_base import TestCaseBase, strict_mode_guard

import sot
from sot.psdb import check_no_breakgraph


def double_num(num: float | int):
    return num * 2


def double_num_with_breakgraph(num: float | int):
    sot.psdb.breakgraph()
    return num * 2


@check_no_breakgraph
def test_map_list(x: list):
    return list(map(double_num, x))


@check_no_breakgraph
def test_map_list_comprehension(x: list):
    return [i for i in map(double_num, x)]  # noqa: C416


@check_no_breakgraph
def test_map_tuple(x: tuple):
    return tuple(map(double_num, x))


@check_no_breakgraph
def test_map_tuple_comprehension(x: tuple):
    return [i for i in map(double_num, x)]  # noqa: C416


@check_no_breakgraph
def test_map_range(x: Iterable):
    return list(map(double_num, x))


@check_no_breakgraph
def test_map_range_comprehension(x: Iterable):
    return [i for i in map(double_num, x)]  # noqa: C416


def add_dict_prefix(key: str):
    return f"dict_{key}"


@check_no_breakgraph
def test_map_dict(x: dict):
    return list(map(add_dict_prefix, x))


@check_no_breakgraph
def test_map_dict_comprehension(x: dict):
    return [i for i in map(add_dict_prefix, x)]  # noqa: C416


def test_map_list_with_breakgraph(x: list):
    return list(map(double_num_with_breakgraph, x))


@check_no_breakgraph
def test_map_unpack(x: list):
    a, b, c, d = map(double_num, x)
    return a, b, c, d


@check_no_breakgraph
def test_map_for_loop(x: list):
    res = 0
    for i in map(double_num, x):
        res += i
    return res


class TestMap(TestCaseBase):
    def test_map(self):
        self.assert_results(test_map_list, [1, 2, 3, 4])
        self.assert_results(test_map_tuple, (1, 2, 3, 4))
        self.assert_results(test_map_range, range(5))
        self.assert_results(test_map_dict, {"a": 1, "b": 2, "c": 3})

    def test_map_comprehension(self):
        self.assert_results(test_map_list_comprehension, [1, 2, 3, 4])
        self.assert_results(test_map_tuple_comprehension, (1, 2, 3, 4))
        self.assert_results(test_map_range_comprehension, range(5))
        self.assert_results(
            test_map_dict_comprehension, {"a": 1, "b": 2, "c": 3}
        )

    def test_map_with_breakgraph(self):
        with strict_mode_guard(0):
            self.assert_results(test_map_list_with_breakgraph, [1, 2, 3, 4])

    def test_map_unpack(self):
        self.assert_results(test_map_unpack, [1, 2, 3, 4])

    def test_map_for_loop(self):
        self.assert_results(test_map_for_loop, [7, 8, 9, 10])


if __name__ == "__main__":
    unittest.main()
