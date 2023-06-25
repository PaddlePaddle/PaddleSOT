import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: int, y: paddle.Tensor):
    z = 3

    def local(a, b=5):
        return a + x + z + b + y

    return local(4)


class TestExecutor(TestCaseBase):
    def test_closure(self):
        self.assert_results(foo, 1, paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()


# Instructions:
# LOAD_CLOSURE
# LOAD_DEREF
# LOAD_CLASSDEREF
# STORE_DEREF
# DELETE_DEREF
