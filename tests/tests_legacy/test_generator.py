import unittest

from test_case_base import TestCaseBase


def gen():
    yield from range(10)


def case1():
    sum = 0
    for i in gen():
        sum += i
    return sum


class TestGen(TestCaseBase):
    def test_gen(self):
        self.assert_results(case1)


if __name__ == "__main__":
    unittest.main()
