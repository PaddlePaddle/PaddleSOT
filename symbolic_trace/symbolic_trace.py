from .utils import Singleton, NameGenerator
from .statement_ir import StatementIRFactory, Statement

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

    def call_API(self, apiname, inputs, outputs): 
        assert callable(apiname), "call_API must receive a paddle api."
        stmt = Statement("api", apiname, inputs, outputs)
        self.sir_stack[-1].add_statement(stmt)

    def start_compile(self):
        """
        1. inputs, outputs 决定算法 (@xiaojian)
        2. StatementIR -> Program + Program -> Executor(to_static)
        3. call 的处理。小形执行器
        """
        print ("start subgraph compile and execution.")
        print (self.sir_stack[-1])
        #TODO(xiongkun): do program generation and execution.
        pass
