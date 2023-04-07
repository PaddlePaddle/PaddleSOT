from paddle.utils import map_structure
from .statement_ir import Symbol
import paddle

def replace_symbol(values, state):
    def replace(x):
        if isinstance(x, Symbol):
            return state[x.name]
        return x
    return map_structure(replace, values)

def set_symbol(outs, out_symbols, state):
    def _set(out, sym):
        if isinstance(sym, Symbol): 
            state[sym.name] = out
    map_structure(_set, outs, out_symbols)

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
            set_symbol(outs, stmt.outputs, state)
        # fetch outputs
        return replace_symbol(SIR.outputs, state)

    def call(self, stmt, inputs):
        SIR = self.get_sir(name)
        state = {}
        for inp, inp_sym in zip(inputs, SIR.inputs): 
            state[inp_sym.name] = inp
        return run_sir(stmt.name, state)

    def api(self, stmt, inputs):
        return stmt.name(*inputs)
        
    def method(self, stmt, inputs):
        var = inputs[0]
        return getattr(var, stmt.name)(*inputs[1:])

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

