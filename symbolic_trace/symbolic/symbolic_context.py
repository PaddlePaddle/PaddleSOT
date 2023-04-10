from __future__ import annotations

import inspect
from typing import Any

from ..utils import Singleton, NameGenerator, no_eval_frame, log, is_proxy_tensor
from .statement_ir import StatementIRFactory, Statement, Symbol
from .interpreter import compile_sir
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
        self.cached_SIR = {}            # { name : (SIR_name, input_hash, all_output) }
                                        # SIR.input and SIR.output do not contain values which is not ProxyTensor
                                        # but, all inputs should be a part of the cache key
                                        # and we need return a complete output

    def __enter__(self):
        self.reset()
        self.sir_stack.append(self.statement_factory.create())
        self.under_dy2static = True

    def __exit__(self, type, value, traceback):
        self.under_dy2static = False

    def new_varname(self):
        return self.var_name_generator.next()

    @no_eval_frame
    # should generate a unique name for every funtion
    def frame_enter(self, name, inputs):
        breakpoint()

        # need a better hash strategy
        key_set = set()
        for inp in paddle.utils.flatten(inputs):
            if is_proxy_tensor(inp):
                key_set.add(inp.meta)
            else:
                key_set.add(inp)

        cur_key = hash(key_set)

        if name in self.cached_SIR.keys():
            sir_name, input_hash, outs = self.cached_SIR[name]

            if cur_key == hashkey:
                cur_sir = StatementIRFactory()[sir_name]
                self.call_SIR(cur_sir, cur_sir.inputs, cur_sir.outputs)
                return True

        new_sir = self.statement_factory.create()
        setattr(new_sir, "func_name", name)
        setattr(new_sir, "input_hash", cur_key)
        self.sir_stack.append(new_sir)
        return None

    @no_eval_frame
    def frame_leave(self, output):
        breakpoint()
        cur_sir = self.sir_stack[-1]
        cur_sir.analysis_inputs()
        inputs_symbols = cur_sir.inputs
        flat_outputs = paddle.utils.flatten(output)
        outputs_symbols = [Symbol(output.name) for output in flat_outputs if is_proxy_tensor(output)]\
        outs = 
        self.cached_SIR[cur_sir.func_name] = (cur_sir.name, cur_sir.input_hash, outs)
        self.sir_stack.pop()

        self.call_SIR(cur_sir, inputs_symbols, outputs_symbols)
        return 

    def call_SIR(self, sirname, inputs, outputs): 
        stmt = Statement("call", sirname, inputs, outputs)
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
    
    def reset_TOS(self):
        self.sir_stack.pop()
        self.sir_stack.append(self.statement_factory.create())

    @no_eval_frame
    def start_return(self, runtime_context, output: Any, is_return: bool = False): 
        """ 
        start compile and return the python function, which must can be to_static without errors.
        """
        self.start_compile(runtime_context, output, is_return=True)
        return paddle.utils.map_structure(lambda x: x.value() if is_proxy_tensor(x) else x, output)

    @no_eval_frame
    def start_compile(self, runtime_context, output: Any, is_return: bool = False):
        cur_sir = self.sir_stack[-1]

        # step0: if no statement, do nothing and return.
        if len(cur_sir.statements) == 0: 
            return

        # step1: analysis sir inputs and outputs
        cur_sir.analysis_inputs()

        flat_outputs = paddle.utils.flatten(output)
        outputs_symbols = [Symbol(output.name) for output in flat_outputs if is_proxy_tensor(output)]
        if len(outputs_symbols) == 0: 
            return

        if is_return:
            cur_sir.outputs = outputs_symbols
        else:
            # TODO(SigureMo): Automatically get the calling frame
            current_frame = inspect.currentframe()
            assert current_frame is not None
            calling_frame = current_frame
            while calling_frame.f_code.co_name != "case1": # TODO: As above
                calling_frame = calling_frame.f_back
            assert calling_frame is not None
            cur_sir.analysis_outputs(runtime_context, calling_frame, additional_outputs=outputs_symbols)

        log (1, "start subgraph compile and execution.\n")
        log (1, self.sir_stack[-1], '\n')

        # step2: call compile_sir and get python function
        py_func = compile_sir(self, cur_sir.name)

        # step3: construct inputs
        to_static_inputs = construct_eager_inputs(cur_sir.inputs, runtime_context.get_runtime())

        # step4: execute to_static and get outputs
        eager_tensor_outputs = paddle.jit.to_static(py_func)(to_static_inputs)

        # step5: reset runtime_value and proxytensor.
        for symbol, eager_tensor_output in zip(cur_sir.outputs, eager_tensor_outputs):
            runtime_context.runtime_name_to_proxy_tensor[symbol.name].set_value(eager_tensor_output)

        # step6: GC and reset TOS
        self.reset_TOS()

def construct_eager_inputs(input_names, runtime_value): 
    state = []
    for inp in input_names: 
        assert runtime_value[inp.name].value() is not None, "Inputs of graph must have value."
        state.append(runtime_value[inp.name].value())
    return state
