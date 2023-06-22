from __future__ import annotations

import inspect
import operator
import unittest

from test_case_base import TestCaseBase

import paddle
from sot.opcode_translator.executor.function_graph import FunctionGraph
from sot.opcode_translator.executor.tracker import DanglingTracker, LocalTracker
from sot.opcode_translator.executor.variables import (
    BuiltinVariable,
    VariableFactory,
)


def compute(x, y):
    ret = BuiltinVariable(operator.add, x.graph, DanglingTracker())(x, y)
    return BuiltinVariable(operator.mul, x.graph, DanglingTracker())(ret, x)


def try_add(x, y):
    return BuiltinVariable(operator.add, x.graph, DanglingTracker())(x, y)


class TestRollback(TestCaseBase):
    def test_rollback(self):
        frame = inspect.currentframe()
        graph = FunctionGraph(frame)
        a = paddle.to_tensor(1.0)
        b = paddle.to_tensor(2.0)
        a = VariableFactory().from_value(a, graph, LocalTracker("a"))
        b = VariableFactory().from_value(b, graph, LocalTracker("b"))
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
