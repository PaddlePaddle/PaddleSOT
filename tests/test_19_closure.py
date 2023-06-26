import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: int, y: paddle.Tensor):
    z = 3

    def local(a, b=5):
        return a + x + z + b + y

    return local(4)


def foo2(y: paddle.Tensor, x=1):
    z = 3

    def local(a, b=5):
        return a + x + z + b + y

    return local(4)


def foo3(y: paddle.Tensor, x=1):
    z = 3

    def local(a, b=5):
        nonlocal z
        z = 4
        return a + x + z + b + y

    return local(4)


global_z = 3


def foo4(y: paddle.Tensor):
    def local(a, b=5):
        global global_z
        global_z = 4
        return a + global_z + b + y

    return local(1)


class TestExecutor(TestCaseBase):
    def test_closure(self):
        self.assert_results(foo, 1, paddle.to_tensor(2))
        self.assert_results(foo2, paddle.to_tensor(2))
        self.assert_results(foo3, paddle.to_tensor(2))
        # TODO(SigureMo) SideEffects have not been implemented yet, we need to skip them
        # self.assert_results(foo4, paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()


# Instructions:
# LOAD_CLOSURE
# LOAD_DEREF
# LOAD_CLASSDEREF
# STORE_DEREF
# DELETE_DEREF
# STORE_GLOBAL
