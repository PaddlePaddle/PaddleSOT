from __future__ import annotations

import unittest

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle


def foo(x: list[paddle.Tensor], y: list[paddle.Tensor]):
    return x[0] + y[0]


def bar(x: list[paddle.Tensor], y: int, z: int):
    return x[y + z] + 1


class TestTraceListArg(TestCaseBase):
    def test_foo(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        c = paddle.to_tensor([3, 4])

        with test_instruction_translator_cache_context() as cache:
            self.assert_results(foo, [a], [b])
            self.assertEqual(cache.translate_count, 1)
            self.assert_results(foo, [b], [a])  # Cache hit
            self.assertEqual(cache.translate_count, 1)
            self.assert_results(foo, [a], [c])  # Cache miss
            self.assertEqual(cache.translate_count, 2)

    def test_bar(self):
        a = [paddle.to_tensor(1), paddle.to_tensor(2), paddle.to_tensor(3)]
        b = [paddle.to_tensor([2, 3]), paddle.to_tensor(4), paddle.to_tensor(5)]

        with test_instruction_translator_cache_context() as cache:
            self.assert_results(bar, a, 1, 1)
            self.assertEqual(cache.translate_count, 1)
            self.assert_results(bar, a, 2, 0)  # Cache miss
            self.assertEqual(cache.translate_count, 2)
            self.assert_results(bar, b, 1, 1)  # Cache hit
            self.assertEqual(cache.translate_count, 3)


if __name__ == "__main__":
    unittest.main()
