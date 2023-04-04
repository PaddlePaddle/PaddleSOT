from .utils import Singleton, NameGenerator
from .statement_ir import StatementIRFactory, Statement, Symbol
from .interpreter import run_sir, compile_sir, compile_ast_modify
import paddle

@Singleton
class SymbolicTraceContext:
    def __init__(self):
        self.reset()

    def reset(self):
        self.statement_factory = StatementIRFactory()
        self.var_name_generator = NameGenerator("var_")
        self.sir_stack = []
        self.under_dy2static = None

    def __enter__(self):
        self.reset()
        self.frame_enter()
        self.under_dy2static = True

    def __exit__(self, type, value, traceback):
        self.under_dy2static = False

    def new_varname(self):
        return self.var_name_generator.next()

    def frame_enter(self):
        self.sir_stack.append(self.statement_factory.create())
    
    def call_SIR(self, sirname, inputs, outputs): 
        stmt = Statment("call", sirname, inputs, outputs)
        self.sir_stack[-1].add_statement(stmt)

    def call_API(self, api, inputs, outputs): 
        assert callable(api), "call_API must receive a paddle api."
        stmt = Statement("api", api, inputs, outputs)
        self.sir_stack[-1].add_statement(stmt)

    def call_METHOD(self, method_name, inputs, outputs): 
        assert isinstance(method_name, str), "call_METHOD must method api name. string."
        assert isinstance(inputs[0], Symbol), "call_METHOD must first augument must be Symbol Variable."
        stmt = Statement("method", method_name, inputs, outputs)
        self.sir_stack[-1].add_statement(stmt)

    def get_sir(self, name):
        return self.statement_factory[name]

    def start_compile(self, runtime_value):
        """ 
        start compile and return the python function, which must can be to_static without errors.
        """
        print ("start subgraph compile and execution.")

        cur_sir = self.sir_stack[-1]
        # step1: analysis sir inputs and outputs  (@xiaojian)
        cur_sir.inputs = [ Symbol('var_0') ]
        cur_sir.outputs = Symbol('var_4')
        print (self.sir_stack[-1])

        # step2: call compile_sir and get python function
        py_func = compile_sir(cur_sir.name)

        # step3: construct inputs
        to_static_inputs = construct_eager_inputs(cur_sir, runtime_value)

        # step4: execute to_static and get outputs
        if len(cur_sir.statements) == 0: 
            return 
        outputs = paddle.jit.to_static(compile_ast_modify)(cur_sir.name, to_static_inputs)

        # step5: reset runtime_value and proxytensor.
        return outputs

def construct_eager_inputs(SIR, runtime_value): 
    state = []
    for inp in SIR.inputs: 
        assert runtime_value[inp.name].value() is not None, "Inputs of graph must have value."
        state.append(runtime_value[inp.name].value())
    return state
