"""
THIS FILE IS PRIVATE !!

use interface in symbolic_trace.py first.
"""

from .utils import Singleton, NameGenerator
from paddle.utils import is_sequence, map_structure

class Symbol: 
    """ 
    we need this class to distinguish the string and `math variable`
    """
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

class Statement:
    def __init__(self, type, name, inputs, outputs):
        assert type in ['call', 'api', 'method']
        self.name = name
        self.inputs = inputs # list of Symbol | PythonObj
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

    def analysis_inputs(self): 
        pass

    def analysis_outputs(self):
        pass

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

class StatementIRFactory:
    def __init__(self):
        self.cache = {}
        self.name_generator = NameGenerator("SIR_")

    def __getitem__(self, key):
        return self.cache[key]

    def create(self):
        name = self.name_generator.next()
        sir = StatementIR(name)
        self.cache[name] = sir
        return sir

