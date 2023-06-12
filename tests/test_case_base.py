import contextlib
import unittest

import numpy as np

import paddle
from sot import symbolic_translate
from sot.opcode_translator.executor.opcode_executor import (
    InstructionTranslatorCache,
)


@contextlib.contextmanager
def test_instruction_translator_cache_context():
    cache = InstructionTranslatorCache()
    cache.clear()
    yield cache
    cache.clear()


class TestCaseBase(unittest.TestCase):
    def assert_nest_match(self, x, y):
        cls_x = type(x)
        cls_y = type(y)
        self.assertIs(
            cls_x, cls_y, msg=f"type mismatch, x is {cls_x}, y is {cls_y}"
        )
        container_types = (tuple, list, dict, set)
        if cls_x in container_types:
            self.assertEqual(
                len(x),
                len(y),
                msg=f"length mismatch, x is {len(x)}, y is {len(y)}",
            )
            if cls_x in (tuple, list):
                for x_item, y_item in zip(x, y):
                    self.assert_nest_match(x_item, y_item)
            elif cls_x is dict:
                for x_key, y_key in zip(x.keys(), y.keys()):
                    self.assert_nest_match(x_key, y_key)
                    self.assert_nest_match(x[x_key], y[y_key])
            elif cls_x is set:
                # TODO: Nested set is not supported yet
                self.assertEqual(x, y)
        elif cls_x in (np.ndarray, paddle.Tensor):
            np.testing.assert_allclose(x, y)
        else:
            self.assertEqual(x, y)

    def assert_results(self, func, *inputs):
        sym_output = symbolic_translate(func)(*inputs)
        paddle_output = func(*inputs)
        self.assert_nest_match(sym_output, paddle_output)
