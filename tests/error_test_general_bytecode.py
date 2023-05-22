import dis
import unittest

from test_case_base import TestCaseBase


def func():
    return True is True is not False


def func2(x):
    # TODO(@xiaojian): SIR not used by output.
    y = x + 1
    return True is True is not False


def func3():
    i = 0

    def inner():
        return i + 1

    return inner()


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(func3)
        # self.assert_results(func2, paddle.to_tensor(1.0))


dis.dis(func3)

if __name__ == "__main__":
    unittest.main()
