from __future__ import annotations

import re
import unittest

import paddle
from sot import symbolic_translate


def case1(x):
    return n  # noqa: F821


def case2(x):
    x = x + 1
    return x @ x


def case3(x):
    y = x.undefined_attr
    return y


def case4_inner(x):
    y = x * 2
    print(x)
    y = y + 1
    return y[100]


def case4(x):
    return case4_inner(x)


class TestAnalysisInputs(unittest.TestCase):
    def catch_error(self, func, inputs, error_lines: int | list[int]):
        if isinstance(error_lines, int):
            error_lines = [error_lines]
        try:
            symbolic_translate(func)(inputs)
        except Exception as e:
            print(e)
            match_results = re.compile(r'File ".*", line (\d+)').findall(str(e))
            match_results = list(map(int, match_results))
            assert len(match_results) == len(
                error_lines
            ), f"{len(match_results)} is not equal {len(error_lines)}"
            assert (
                match_results == error_lines
            ), f"{match_results} is not equal {error_lines}"

    def test_all_case(self):
        self.catch_error(case1, paddle.rand([2, 1]), 11)
        # TODO: support runtime error
        # self.catch_error(case2, paddle.rand([2, 1]))
        self.catch_error(case3, paddle.rand([2, 1]), 20)
        self.catch_error(case4, paddle.rand([2, 1]), [32, 28])


if __name__ == "__main__":
    unittest.main()
