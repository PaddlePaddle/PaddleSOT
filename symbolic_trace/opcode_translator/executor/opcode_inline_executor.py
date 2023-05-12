from ...utils import log
from .opcode_executor import OpcodeExecutorBase, Stop
from .tracker import Tracker


class FunctionGlobalTracker:
    pass


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
        # prepare locals for args
        argc = self._code.co_argcount
        arg_names = self._code.co_varnames[:argc]
        for name, value in zip(arg_names, args):
            from .variables import VariableTracker

            assert isinstance(value, VariableTracker)
            self._locals[name] = value

        # prepare locals for kwargs
        for name, value in kwargs.items():
            from .variables import VariableTracker

            assert isinstance(value, VariableTracker)
            self._locals[name] = value

        log(
            5, f"[INLINE CALL] {self._code.co_name} with locals: ", self._locals
        )

    def _prepare_virtual_env(self):
        # prepare globals
        from .variables import VariableTrackerFactory

        for idx, (name, value) in enumerate(self._fn_value.__globals__.items()):
            self._globals[name] = VariableTrackerFactory.from_value(
                value, self._graph, FunctionGlobalTracker()
            )

        # prepare consts
        for value in self._code.co_consts:
            self._co_consts.append(
                VariableTrackerFactory.from_value(
                    value, self._graph, FunctionConstTracker(value)
                )
            )

    def RETURN_VALUE(self, instr):
        self.return_value = self.pop()
        return Stop()

    def inline_call(self):
        self.run()
        return self.return_value
