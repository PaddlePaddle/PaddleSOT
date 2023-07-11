from __future__ import annotations

import unittest

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle


def test_guard_fn(fn, inp):
    if fn is None:
        return 0
    else:
        return fn(inp)


class TestGuardOutputs(TestCaseBase):
    def test_non_operator_related_fn(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                test_guard_fn,
                paddle.nn.functional.relu,
                paddle.to_tensor([1.0, -1.0]),
            )
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(
                test_guard_fn,
                paddle.nn.functional.gelu,
                paddle.to_tensor([1.0, -1.0]),
            )
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(
                test_guard_fn,
                paddle.nn.functional.relu,
                paddle.to_tensor([-1.0, -1.0]),
            )
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(
                test_guard_fn, None, paddle.to_tensor([-1.0, -1.0])
            )
            self.assertEqual(ctx.translate_count, 3)

        deleted_cnt = 0

        class Callable:
            def __call__(self, var):
                return paddle.nn.functional.relu(var)

            def __del__(self):
                nonlocal deleted_cnt
                deleted_cnt += 1

        fn1 = Callable()
        fn2 = Callable()
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                test_guard_fn, fn1, paddle.to_tensor([1.0, -1.0])
            )
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(
                test_guard_fn, fn2, paddle.to_tensor([1.0, -1.0])
            )
            self.assertEqual(ctx.translate_count, 2)
            del fn1
            assert deleted_cnt == 1
            self.assert_results(
                test_guard_fn, fn2, paddle.to_tensor([1.0, -1.0])
            )
            self.assertEqual(ctx.translate_count, 2)


if __name__ == "__main__":
    unittest.main()
