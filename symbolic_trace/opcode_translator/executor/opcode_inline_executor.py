from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from ...utils import log
from .opcode_executor import OpcodeExecutorBase, Stop
from .tracker import ConstTracker, DummyTracker, Tracker

if TYPE_CHECKING:
    from .pycode_generator import PyCodeGen
    from .variables import FunctionVariable


class FunctionGlobalTracker(Tracker):
    def __init__(self, func: FunctionVariable, name: str):
        super().__init__([func])
        self.func = func
        self.name = name
        # TODO: handle builtins

    def gen_instructions(self, codegen: PyCodeGen):
        self.func.tracker.gen_instructions(codegen)
        codegen.gen_load_attr("__globals__")
        codegen.gen_load_const(self.name)
        codegen.gen_subscribe()

    def trace_value_from_frame(self):
        return lambda frame: self.func.tracker.trace_value_from_frame()(
            frame
        ).__globals__[self.name]


class FunctionConstTracker(Tracker):
    def __init__(self, value):
        super().__init__([])
        self.value = value


class OpcodeInlineExecutor(OpcodeExecutorBase):
    def __init__(self, fn_variable, *args, **kwargs):
        self._fn_var = fn_variable
        self._fn_value = fn_variable.value
        self.return_value = None
        super().__init__(fn_variable.value.__code__, fn_variable.graph)
        self._prepare_locals(*args, **kwargs)
        # TODO: consider generator.

    def _prepare_locals(self, *args, **kwargs):
        from .variables import VariableTracker, VariableTrackerFactory

        sig = inspect.signature(self._fn_value)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for name, value in bound_args.arguments.items():
            assert name in sig.parameters
            # Convert varargs and kwargs to Variable
            if sig.parameters[name].kind == inspect.Parameter.VAR_POSITIONAL:
                tracker = DummyTracker(value)
            elif sig.parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
                tracker = DummyTracker(list(value.values()))
            # Convert default args to Variable
            elif not isinstance(value, VariableTracker):
                tracker = ConstTracker(value)
            else:
                tracker = value.tracker
            value = VariableTrackerFactory.from_value(
                value, self._graph, tracker
            )
            self._locals[name] = value

        log(
            5, f"[INLINE CALL] {self._code.co_name} with locals: ", self._locals
        )

    def _prepare_virtual_env(self):
        # prepare globals
        from .variables import VariableTrackerFactory

        for name, value in self._fn_value.__globals__.items():
            self._globals[name] = VariableTrackerFactory.from_value(
                value, self._graph, FunctionGlobalTracker(self._fn_var, name)
            )

        # prepare builtins
        # Waiting for https://github.com/2742195759/paddle-symbolic-trace/pull/73
        # for name, value in self._fn_value.__builtins__.items():
        #     self._builtins[name] = VariableTrackerFactory.from_value(
        #         value, self._graph, FunctionGlobalTracker(self._fn_var, name)
        #     )

        # prepare consts
        for value in self._code.co_consts:
            self._co_consts.append(
                VariableTrackerFactory.from_value(
                    value, self._graph, ConstTracker(value)
                )
            )

    def RETURN_VALUE(self, instr):
        self.return_value = self.pop()
        return Stop()

    def inline_call(self):
        self.run()
        return self.return_value
