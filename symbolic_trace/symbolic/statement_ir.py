"""
THIS FILE IS PRIVATE !!

use interface in symbolic_context.py first.
"""
from __future__ import annotations

import types

from ..utils import NameGenerator, log, Singleton
from paddle.utils import is_sequence, map_structure, flatten
import paddle
from .bytecode_analysis import output_analysis

class Symbol: 
    """ 
    we need this class to distinguish the string and `math variable`
    """
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)

class Statement:
    def __init__(self, type, name, inputs, outputs):
        assert type in ['call', 'api', 'method']
        self.name = name
        self.inputs = inputs # (list of Symbols, dict of Symbols) 
        self.outputs = outputs # list of Symbol | PythonObj
        self.type = type

    def __str__(self):
        def to_string(inps):
            if isinstance(inps, str) or not is_sequence(inps):
                return inps.__str__()
            inps = map(lambda x: x.__str__(), inps)
            return ", ".join(inps)
        name = self.name if isinstance(self.name, str) else 'paddle.' + self.name.__name__
        return "%s || %s = %s (%s) " % (self.type + ' '*(10 - len(self.type)), to_string(self.outputs), name, to_string(self.inputs))

    def __repr__(self):
        return self.__str__()

class StatementIR :
    """
    Don't create by yourself, just use the StatementIRCache.get()
    """
    def __init__(self, name):
        self.name = name
        self.inputs = [] # list of Symbol | PythonObj
        self.outputs = [] # list of Symbol | PythonObj
        self.statements = [] # list of Statement
        pass

    def add_input(self, input):
        self.inputs.append(input)

    def add_output(self, output):
        self.outputs.append(output)

    def add_statement(self, statement):
        assert isinstance(statement, Statement)
        self.statements.append(statement)

    def analyse_inputs(self): 
        used_symbols = set()
        generated_symbols = set()
        for stmt in self.statements:
            for inp in flatten(stmt.inputs):
                if isinstance(inp, Symbol):
                    used_symbols.add(inp)
            for out in flatten(stmt.outputs):
                if isinstance(out, Symbol):
                    generated_symbols.add(out)                

        input_symbols = list(used_symbols - generated_symbols)
        return input_symbols

    def analyse_outputs(self, runtime_context, user_frames: list[types.FrameType], additional_outputs=[]):
        reads_symbols = []
        for frame in user_frames:
            log(2, f"[analyse_outputs] frame name is `{frame.f_code.co_name}`", "\n")
            reads = output_analysis(frame)
            log(2, f"[analyse_outputs] reads is `{reads}`", "\n")
            reads_locals = [frame.f_locals[name] for name in reads]
            for local in reads_locals:
                proxy_tensor = local
                if isinstance(local, paddle.Tensor):
                    proxy_tensor = runtime_context.from_tensor(local)
                # TODO(SigureMo): Handle other types
                reads_symbols.append(Symbol(proxy_tensor.name))

        # Add additional outputs
        output_symbols = set(reads_symbols) | set(additional_outputs)

        # Remove the outputs that are not in the statements
        statement_output_symbols = {
            out
            for stmt in self.statements
            for out in paddle.utils.flatten(stmt.outputs)
            if isinstance(out, Symbol)
        }
        output_symbols = output_symbols & statement_output_symbols
        return list(output_symbols), list(reads_symbols)


    def __str__(self):
        strs = []
        strs.append("StatmentIR: %s" % self.name)
        strs.append("  inputs: %s" % map_structure(lambda x: x.name, self.inputs))
        strs.append("  outputs: %s" % map_structure(lambda x: x.name, self.outputs))
        strs.append("  statements: ")
        for stmt in self.statements:
            strs.append("    %s" % stmt)
        return "\n".join(strs)

    def __repr__(self):
        return self.__str__()

@Singleton
class StatementIRFactory:
    def __init__(self):
        self.cache = {}
        self.name_generator = NameGenerator("SIR_")

    def __getitem__(self, key):
        return self.cache[key]

    def create(self, input_name=None):
        if input_name:
            name = input_name
        else:
            name = self.name_generator.next()

        sir = StatementIR(name)
        self.cache[name] = sir
        return sir
    
    def clear(self):
        want_clear = [key for key in self.cache.keys() if key.startswith("SIR_")]
        for key in want_clear:
            del self.cache[key]

        self.name_generator = NameGenerator("SIR_")



@Singleton
class SIRRuntimeCache:
    def __init__(self):
        self.cache = {}         #    { name : (inputs, outputs, free_vars) }
                                #       inputs  : can be used when call_SIR, if free_vars exist
                                #       outputs : used for generator new ProxyTensor output before fallback
                                #       free_vars: (name, function)

    def __getitem__(self, key):
        return self.cache[key]
    
    def has_key(self, key):
        return key in self.cache.keys()

    def set_origin_inputs(self, key, inputs):
        if key in self.cache.keys():
            val = self.cache[key]
            self.cache[key] = (inputs, val[1], val[2])
        else:
            self.cache[key] = (inputs, None, None)

    def set_origin_outputs(self, key, outputs):
        if key in self.cache.keys():
            val = self.cache[key]
            self.cache[key] = (val[0], outputs, val[2])
        else:
            self.cache[key] = (None, outputs, None)

    def set_free_vars(self, key, free_vars):
        if key in self.cache.keys():
            val = self.cache[key]
            self.cache[key] = (val[0], val[1], free_vars)
        else:
            self.cache[key] = (None, None, free_vars)

    def get_origin_inputs(self, key):
        if key in self.cache.keys():
            return self.cache[key][0]
        else:
            return None

    def get_origin_outputs(self, key):
        if key in self.cache.keys():
            return self.cache[key][1]
        else:
            return None

    def get_free_vars(self, key):
        if key in self.cache.keys():
            return self.cache[key][2]
        else:
            return None
