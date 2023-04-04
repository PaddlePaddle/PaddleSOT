from paddle.utils import map_structure
from .statement_ir import Symbol

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

def run_sir(name, state): 
    from symbolic_trace.symbolic_trace import SymbolicTraceContext
    SIR = SymbolicTraceContext().get_sir(name)
    gc_pass(SIR)
    for stmt in SIR.statements:
        inputs = replace_symbol(stmt.inputs, state)
        outs = getattr(Instruction, stmt.type)(stmt, inputs)
        set_symbol(outs, stmt.outputs, state)
    # fetch outputs
    return replace_symbol(SIR.outputs, state)

def compile_sir(name):
    def wrapper(args):
        """
        This function will be decorated by paddle.to_static.
        so the args is variables, not eager tensors.
        """
        from symbolic_trace.symbolic_trace import SymbolicTraceContext
        SIR = SymbolicTraceContext().get_sir(name)
        state = {}
        for inp, arg in zip(SIR.inputs, args):
            state[inp.name] = arg
        return run_sir(name, state)
    return wrapper

def compile_ast_modify(name, args):
    """
    因为动转静AST转写不支持闭包，所以会报错。这里我们直接把闭包的内容展开。先临时修复
    This function will be decorated by paddle.to_static.
    so the args is variables, not eager tensors.
    """
    from symbolic_trace.symbolic_trace import SymbolicTraceContext
    SIR = SymbolicTraceContext().get_sir(name)
    state = {}
    for inp, arg in zip(SIR.inputs, args):
        state[inp.name] = arg
    return run_sir(name, state)

def gc_pass(sir):
    pass

class Instruction:
    @staticmethod
    def call(stmt, inputs):
        from symbolic_trace.symbolic_trace import SymbolicTraceContext
        SIR = SymbolicTraceContext().get_sir(name)
        state = {}
        for inp, inp_sym in zip(inputs, SIR.inputs): 
            state[inp_sym.name] = inp
        return run_sir(stmt.name, state)

    @staticmethod
    def api(stmt, inputs):
        return stmt.name(*inputs)
        
    @staticmethod
    def method(stmt, inputs):
        var = inputs[0]
        return getattr(var, stmt.name)(*inputs[1:])

    @staticmethod
    def delete(stmt, inputs):
        """
        for gc !!
        """
        pass

