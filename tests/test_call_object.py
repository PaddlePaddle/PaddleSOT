import unittest

from test_case_base import TestCaseBase

import paddle

patched = lambda self, x: x * self.a

patched2 = lambda self, x: x * self.a + 3


class A:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return self.add(x)

    def add(self, x):
        return x + self.a

    multi = patched


class B:
    def __init__(self, a):
        self.a = A(a)

    def __call__(self, x, func):
        return getattr(self.a, func)(x)

    def self_call(self, x, func):
        return getattr(self.a, func)(self.a, x)


def foo_1(a, x):
    return a(x)


def foo_2(a, x):
    return a.multi(x)


def foo_3(b, x):
    return b(x, "multi")


def foo_4(b, x):
    return b(x, "add")


def foo_5(b, x):
    return b.self_call(x, "multi")


class TestExecutor(TestCaseBase):
    def test_simple(self):
        c = B(13)
        c.a.multi = patched2
        self.assert_results(foo_1, A(13), paddle.to_tensor(2))
        self.assert_results(foo_2, A(13), paddle.to_tensor(2))
        self.assert_results(foo_3, B(13), paddle.to_tensor(2))
        self.assert_results(foo_4, B(13), paddle.to_tensor(2))
        self.assert_results(foo_5, c, paddle.to_tensor(2))
        self.assert_results(foo_4, c, paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()
