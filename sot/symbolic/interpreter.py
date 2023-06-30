from __future__ import annotations

from typing import TYPE_CHECKING

import paddle

from ..utils import map_if
from .statement_ir import SIRRuntimeCache, Symbol

if TYPE_CHECKING:
    from .statement_ir import Statement, StatementIR
    from .symbolic_context import SymbolicTraceContext


def replace_symbol(values: list[Symbol], state: dict[str, Symbol]):
    """
    Replaces Symbol objects in a list with their corresponding values in a state dict.

    Args:
        values: A list of values that may contain Symbol objects.
        state: A dict mapping Symbol names to their corresponding values.

    Returns:
        A new list with Symbol objects replaced by their corresponding values in the state dict.
    """
    return map_if(
        values,
        pred=lambda x: isinstance(x, Symbol),
        true_fn=lambda x: state[x.name],
        false_fn=lambda x: x,
    )


class Interpreter:
    """
    Interpreter is used to interpret and execute SIR.
    """

    def __init__(self, symbolic_context: SymbolicTraceContext):
        self._context = symbolic_context

    def get_sir(self, name: str) -> StatementIR:
        """
        Get the StatementIR object for a given name.

        Args:
            name: The name of the StatementIR.

        Returns:
            The StatementIR object with the given name.
        """
        return self._context.get_sir(name)

    def run_sir(self, name: str, state: dict[str, Symbol]):
        """
        Runs the StatementIR with the given name using the provided state.

        Args:
            name: The name of the given StatementIR to run.
            state: A dict mapping Symbol names to their corresponding values.

        Returns:
            A list of the Symbol of the StatementIR after execution.
        """
        SIR = self.get_sir(name)
        gc_pass(SIR)
        for stmt in SIR.statements:
            inputs = replace_symbol(stmt.inputs, state)
            outs = getattr(self, stmt.type)(stmt, inputs)

            def _set(v, s):
                state[s.name] = v

            map_if(
                outs,
                stmt.outputs,
                pred=lambda v, s: isinstance(s, Symbol),
                true_fn=lambda v, s: _set(v, s),
                false_fn=lambda v, s: None,
            )
        # fetch outputs
        return replace_symbol(SIR.outputs, state)

    def call(self, stmt: Statement, inputs):
        SIR = self.get_sir(stmt.name)
        state = prepare_state(SIR, inputs)
        return self.run_sir(stmt.name, state)

    def api(self, stmt, inputs):
        args, kwargs = inputs
        return stmt.name(*args, **kwargs)

    def method(self, stmt, inputs):
        args, kwargs = inputs
        var = args[0]
        return getattr(var, stmt.name)(*args[1:], **kwargs)

    def layer(self, stmt, inputs):
        args, kwargs = inputs
        layer, args = args[0], args[1:]
        return layer(*args, **kwargs)

    def delete(self, stmt, inputs):
        pass


# NOTE(dev): What's it for?
def gc_pass(sir):
    pass


def compile_sir(context: SymbolicTraceContext, name: str):
    """
    Compile a SIR to a new function

    Args:
        context: The context to compile
        name: The name of the sir to compile

    """

    @paddle.jit.not_to_static
    def wrapper(args):
        """
        This function will be decorated by paddle.to_static.
        so the args is variables, not eager tensors.
        """
        interpreter = Interpreter(context)
        SIR = interpreter.get_sir(name)
        state = prepare_state(SIR, args)
        return interpreter.run_sir(name, state)

    return wrapper


def prepare_state(SIR, inputs):
    state = {}

    # update free vars if exsits
    if SIRRuntimeCache().has_key(SIR.name):
        free_var_seeker = SIRRuntimeCache().get_free_vars(SIR.name)
        if free_var_seeker:
            state = free_var_seeker()

    # bind inputs
    for sir_inp, inp in zip(SIR.inputs, inputs):
        state[sir_inp.name] = inp

    return state
