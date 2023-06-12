from __future__ import annotations

import inspect
import unittest

from test_case_base import TestCaseBase

import paddle
from sot.opcode_translator.executor.function_graph import FunctionGraph
from sot.opcode_translator.executor.tracker import DummyTracker
from sot.opcode_translator.executor.variables import VariableFactory


def compute(x, y):
    ret = x + y
    return ret * x


def try_add(x, y):
    return x + y


class TestRollback(TestCaseBase):
    def test_rollback(self):
        frame = inspect.currentframe()
        graph = FunctionGraph(frame)
        a = paddle.to_tensor(1.0)
        b = paddle.to_tensor(2.0)
        a = VariableFactory().from_value(a, graph, DummyTracker)
        b = VariableFactory().from_value(b, graph, DummyTracker)
        out = compute(a, b)
        original_length = len(graph.sir_ctx.TOS.statements)
        memo = graph.save_memo()
        try_add(out, out)

        assert len(graph.sir_ctx.TOS.statements) != len(
            memo.stmt_ir.statements
        ), "After add, we must statement IR."
        graph.restore_memo(memo)

        assert len(graph.sir_ctx.TOS.statements) == original_length


if __name__ == "__main__":
    unittest.main()
