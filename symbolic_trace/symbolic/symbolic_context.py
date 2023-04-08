from __future__ import annotations

import inspect
from typing import Any

from ..utils import Singleton, NameGenerator, no_eval_frame, log, is_proxy_tensor
from .statement_ir import StatementIR, StatementIRFactory, Statement, Symbol
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
        assert isinstance(inputs[0][0], Symbol), "call_METHOD must first augument must be Symbol Variable."
        stmt = Statement("method", method_name, inputs, outputs)
        self.sir_stack[-1].add_statement(stmt)

    def get_sir(self, name):
        return self.statement_factory[name]
    
    def reset_TOS(self):
        self.sir_stack.pop()
        self.sir_stack.append(self.statement_factory.create())

    @no_eval_frame
    def start_return(self, runtime_context, output: Any): 
        """ 
        start compile and return the python function, which must can be to_static without errors.
        """
        self.start_compile(runtime_context, output)
        return paddle.utils.map_structure(lambda x: x.value() if is_proxy_tensor(x) else x, output)

    @no_eval_frame
    def start_compile(self, runtime_context, output: Any):
        cur_sir:StatementIR = self.sir_stack[-1]

        # step0: if no statement, do nothing and return.
        if len(cur_sir.statements) == 0: 
            return

        # step1: analysis sir inputs and outputs
        cur_sir.analysis_inputs()

        flat_outputs = paddle.utils.flatten(output)
        outputs_symbols = [Symbol(output.name) for output in flat_outputs if is_proxy_tensor(output)]
        if len(outputs_symbols) == 0: 
            return

        calling_frame = self.find_user_defined_func_frame()
        cur_sir.analysis_outputs(runtime_context, calling_frame, additional_outputs=outputs_symbols)

        log (1, "start subgraph compile and execution.\n")
        log (1, self.sir_stack[-1], '\n')

        # step2: call compile_sir and get python function
        py_func = compile_sir(self, cur_sir.name)

        # step3: construct inputs
        to_static_inputs = construct_eager_inputs(cur_sir.inputs, runtime_context.get_runtime())

        # step4: execute to_static and get outputs, and clear the name of eager tensor.
        eager_tensor_outputs = paddle.jit.to_static(py_func)(to_static_inputs)
        clear_eager_tensor_name(eager_tensor_outputs)
        log(5, "Input", cur_sir.inputs, to_static_inputs)
        log(5, paddle.jit.to_static(py_func).get_concrete_program(to_static_inputs)[1].train_program)
        log(5, "Output", cur_sir.outputs, eager_tensor_outputs)

        # step5: reset runtime_value and proxytensor.
        for symbol, eager_tensor_output in zip(cur_sir.outputs, eager_tensor_outputs):
            runtime_context.runtime_name_to_proxy_tensor[symbol.name].set_value(eager_tensor_output)

        # step6: GC and reset TOS
        self.reset_TOS()

    def find_user_defined_func_frame(self):
        # TODO(SigureMo): Find a better way to automatically get the calling frame
        current_frame = inspect.currentframe()
        assert current_frame is not None
        calling_frame = current_frame
        calling_stack = []
        while calling_frame.f_back is not None:
            calling_stack.append((calling_frame.f_code.co_name, calling_frame))
            calling_frame = calling_frame.f_back

        calling_stack = list(reversed(calling_stack))

        for frame_idx, (frame_name, _) in enumerate(calling_stack):
            if frame_name == "no_eval_frame_func":
                break

        calling_frame = calling_stack[frame_idx - 1][1]
        assert calling_frame is not None
        return calling_frame

def clear_eager_tensor_name(output_tensors):
    for output_tensor in output_tensors:
        output_tensor.name = ""

def construct_eager_inputs(input_names, runtime_value): 
    output_list = []
    for inp in input_names: 
        assert runtime_value[inp.name].value() is not None, "Inputs of graph must have value."
        output_list .append(runtime_value[inp.name].value())

    return output_list
