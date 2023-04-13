import paddle
from paddle.utils import map_structure, flatten

from ..utils import map_if, meta_str
from .statement_ir import Symbol, SIRRuntimeCache


def replace_symbol(values, state):
    return map_if(values, 
                  pred=lambda x: isinstance(x, Symbol), 
                  true_fn=lambda x: state[x.name],
                  false_fn=lambda x: x)

class Interpreter: 
    def __init__(self, symbolic_context):
        self._context = symbolic_context

    def get_sir(self, name):
        return self._context.get_sir(name)

    def run_sir(self, name, state):
        SIR = self.get_sir(name)
        gc_pass(SIR)
        for stmt in SIR.statements:
            inputs = replace_symbol(stmt.inputs, state)
            outs = getattr(self, stmt.type)(stmt, inputs)
            def _set(v, s):
                state[s.name] = v
            map_if(outs, stmt.outputs, 
                   pred=lambda v, s: isinstance(s, Symbol),
                   true_fn=lambda v, s: _set(v, s),
                   false_fn=lambda v, s: None)

        # fetch outputs
        return replace_symbol(SIR.outputs, state)

    def call(self, stmt, inputs):
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

    def delete(self, stmt, inputs):
        pass

def gc_pass(sir):
    pass

def compile_sir(context, name):
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