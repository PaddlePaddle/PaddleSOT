from paddle.utils import map_structure, flatten
from ..utils import map_if
from .statement_ir import Symbol
import paddle

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
        SIR = self.get_sir(name)
        state = {}
        for inp, inp_sym in zip(inputs, SIR.inputs): 
            state[inp_sym.name] = inp
        return run_sir(stmt.name, state)

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
        state = {}
        for inp, arg in zip(SIR.inputs, args):
            state[inp.name] = arg
        return interpreter.run_sir(name, state)
    return wrapper

