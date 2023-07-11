# GET_ITER (new)
# FOR_ITER (new)

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase, strict_mode_guard

import paddle
from sot import symbolic_translate
from sot.opcode_translator.executor.opcode_executor import (
    InstructionTranslatorCache,
)


def gener():
    yield 1
    yield 2
    yield 3


def for_list_1(x: paddle.Tensor):
    for i in [1, 2, 3]:
        x += i

        if x > 2:
            x += 1
        else:
            x -= 1
    return x


def for_list_2(x: paddle.Tensor):
    for i in [1, 2, 3]:
        x += i

        if i > 2:
            x += 1
        else:
            x -= 1
    return x


def for_dict(x: paddle.Tensor):
    map = {1: 2, 3: 4}
    for k in map.keys():
        x += k

    for v in map.values():
        x += v

    for k, v in map.items():
        x += k
        x += v

    return x


def for_iter(x, it):
    for item in it:
        x += item
    return x


def for_for_fallback(x, it):
    for i in [1, 2, 3]:
        for item in it:
            x += item
    return x


def for_break(x: paddle.Tensor, it):
    for i in [1, 2, 3]:
        x += i
        if i == 2:
            break
    for i in it:
        x += i
        if i == 2:
            break
    return x


def for_continue(x: paddle.Tensor, it):
    for i in [1, 2, 3]:
        if i == 2:
            continue
        x += i

    for i in it:
        if i == 2:
            continue
        x += i
    return x


def for_enumerate_var_with_nested_range(x_array):
    x = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = paddle.fluid.dygraph.to_variable(x_array)
    for i, num in enumerate(x_array):
        for idx in range(num):
            x = x + num
    return x


def for_create_tmp_in_loop(x, it):
    s = x
    for i in it:
        tmp = i
        s += tmp
    return s, tmp


class TestExecutor(TestCaseBase):
    def test_list(self):
        a = paddle.to_tensor(1)
        self.assert_results(for_list_1, a)

    def test_list_with_fallback(self):
        a = paddle.to_tensor(1)
        self.assert_results(for_list_2, a)

    def test_dict(self):
        a = paddle.to_tensor(1)
        self.assert_results(for_dict, a)

    def test_fallback(self):
        a = paddle.to_tensor(1)

        sym_output = symbolic_translate(for_iter)(a, gener())
        paddle_output = for_iter(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    def test_for_for_fallback(self):
        a = paddle.to_tensor(1)

        sym_output = symbolic_translate(for_iter)(a, gener())
        paddle_output = for_iter(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    def test_for_break(self):
        a = paddle.to_tensor(1)
        sym_output = symbolic_translate(for_break)(a, gener())
        paddle_output = for_break(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    def test_for_continue(self):
        a = paddle.to_tensor(1)
        sym_output = symbolic_translate(for_continue)(a, gener())
        paddle_output = for_continue(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    # TODO(zmh): support range for tensor
    # def test_resume_stack(self):
    #     a = [1, 2, 3]
    #     self.assert_results(for_enumerate_var_with_nested_range, a)

    def test_create_var_in_loop(self):
        x = paddle.to_tensor(1, dtype="float32")
        a = [1, 2, 3]
        self.assert_results(for_create_tmp_in_loop, x, a)

        sym_output = symbolic_translate(for_create_tmp_in_loop)(x, iter(a))
        paddle_output = for_create_tmp_in_loop(x, iter(a))
        self.assert_nest_match(sym_output, paddle_output)


def run_list_comp(x):
    out = [s.chunk(2, axis=1) for s in x]
    return out


class TestListComp(TestCaseBase):
    def test_list_comp(self):
        x = [paddle.randn([1, 4]), paddle.randn([1, 4])]
        self.assert_results(run_list_comp, x)


def for_enumerate_cache(func_list, x):
    out = None
    for idx, func in enumerate(func_list):
        out = func(x[idx])
    return out


class TestEnumerateCache(TestCaseBase):
    def test_run(self):
        func_list = [
            paddle.nn.Linear(10, 10),
        ]
        x = [
            paddle.randn([5, 10]),
        ]

        out = symbolic_translate(for_enumerate_cache)(func_list, x)
        out = symbolic_translate(for_enumerate_cache)(func_list, x)
        self.assert_nest_match(InstructionTranslatorCache().translate_count, 1)


if __name__ == "__main__":
    with strict_mode_guard(0):
        unittest.main()
