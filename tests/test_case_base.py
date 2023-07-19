import contextlib
import copy
import inspect
import os
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


def github_action_error_msg(msg: str):
    if 'GITHUB_ACTIONS' in os.environ:
        frame = inspect.currentframe()
        if frame is not None:
            while frame.f_back is not None:
                frame = frame.f_back
            filename = f"tests/{frame.f_code.co_filename[2:]}"
            lineno = frame.f_lineno
            output = f"::error file={filename},line={lineno}::{msg}"
            return output
    return msg


class TestCaseBase(unittest.TestCase):
    def assert_nest_match(self, x, y):
        cls_x = type(x)
        cls_y = type(y)
        msg = github_action_error_msg(
            f"type mismatch, x is {cls_x}, y is {cls_y}"
        )
        self.assertIs(cls_x, cls_y, msg=msg)

        container_types = (tuple, list, dict, set)
        if cls_x in container_types:
            msg = github_action_error_msg(
                f"length mismatch, x is {len(x)}, y is {len(y)}"
            )
            self.assertEqual(
                len(x),
                len(y),
                msg=msg,
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

    def assert_results_with_side_effects(self, func, *inputs):
        sym_inputs = copy.deepcopy(inputs)
        sym_output = symbolic_translate(func)(*sym_inputs)
        paddle_inputs = copy.deepcopy(inputs)
        paddle_output = func(*paddle_inputs)
        self.assert_nest_match(sym_inputs, paddle_inputs)
        self.assert_nest_match(sym_output, paddle_output)


@contextlib.contextmanager
def strict_mode_guard(value):
    if "STRICT_MODE" not in os.environ:
        os.environ["STRICT_MODE"] = "0"
    old_value = os.environ["STRICT_MODE"]
    os.environ["STRICT_MODE"] = str(value)
    yield
    os.environ["STRICT_MODE"] = old_value
