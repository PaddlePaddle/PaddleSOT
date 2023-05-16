import inspect

from ...utils import log
from .opcode_executor import OpcodeExecutorBase, Stop
from .tracker import ConstTracker, DummyTracker, Tracker


class FunctionGlobalTracker(Tracker):
    def __init__(self):
        super().__init__([])


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
