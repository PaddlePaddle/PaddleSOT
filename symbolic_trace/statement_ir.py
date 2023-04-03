"""
THIS FILE IS PRIVATE !!

use interface in symbolic_trace.py first.
"""

from .utils import Singleton, NameGenerator

class Statement:
    def __init__(self, type, name, inputs, outputs):
        assert type in ['call', 'api', 'method']
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.type = type

    def __str__(self):
        def to_string(inps):
            if isinstance(inps, str):
                return inps
            inps = map(lambda x: x if isinstance(x, str) else str(x), inps)
            return ", ".join(inps)
        name = self.name if isinstance(self.name, str) else 'paddle.' + self.name.__name__
        return "%s || %s = %s (%s) " % (self.type, to_string(self.outputs), name, to_string(self.inputs))

    def __repr__(self):
        return self.__str__()

class StatementIR :
    """
    Don't create by yourself, just use the StatementIRCache.get()
    """
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.outputs = []
        self.statements = []
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
        strs.append("  inputs: %s" % self.inputs)
        strs.append("  outputs: %s" % self.outputs)
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

