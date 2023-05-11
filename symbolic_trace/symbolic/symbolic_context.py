from __future__ import annotations

from typing import Any

import paddle

from ..utils import is_proxy_tensor, log, no_eval_frame
from ..utils.frame import find_user_defined_func_frames
from .compile_cache import CompileSIRCache
from .statement_ir import Statement, StatementIR, StatementIRFactory, Symbol


class SymbolicTraceContext:
    def __init__(self):
        self.reset()

    def reset(self):
        self.statement_factory = StatementIRFactory()
        self.statement_factory.clear()
        self.sir_stack = [self.statement_factory.create()]
        # this stack is used for save key of sir, to use at frame_leave
        self.sir_key_stack = []

    @property
    def TOS(self):
        return self.sir_stack[-1]

    def call_SIR(self, sirname, inputs, outputs):
        stmt = Statement("call", sirname, inputs, outputs)
        self.TOS.add_statement(stmt)

    def call_API(self, api, inputs, outputs):
        assert callable(api), "call_API must receive a paddle api."
        stmt = Statement("api", api, inputs, outputs)
        self.TOS.add_statement(stmt)

    def call_METHOD(self, method_name, inputs, outputs):
        assert isinstance(
            method_name, str
        ), "call_METHOD must method api name. string."
        assert isinstance(
            inputs[0][0], Symbol
        ), "call_METHOD must first augument must be Symbol Variable."
        stmt = Statement("method", method_name, inputs, outputs)
        self.TOS.add_statement(stmt)

    def get_sir(self, name):
        return self.statement_factory[name]

    def reset_TOS(self):
        self.sir_stack.pop()
        self.sir_stack.append(self.statement_factory.create())

    @no_eval_frame
    def start_compile(self, runtime_context, output: Any):
        """
        start compile and return the python function, which must can be to_static without errors.
        """
        cur_sir: StatementIR = self.TOS

        # step0: if no statement, do nothing and return.
        if len(cur_sir.statements) == 0:
            return self.fetch_output(output)

        # step1: analyse sir inputs and outputs
        cur_sir.inputs = cur_sir.analyse_inputs()

        flat_outputs = paddle.utils.flatten(output)
        outputs_symbols = [
            Symbol(output.name)
            for output in flat_outputs
            if is_proxy_tensor(output)
        ]
        if len(outputs_symbols) == 0:
            return self.fetch_output(output)

        from ..opcode_translator.skip_translate_names import (
            SKIP_TRANSLATE_NAMES,
        )

        user_frames = find_user_defined_func_frames(
            "symbolic_traced_func", SKIP_TRANSLATE_NAMES
        )
        output_symbols, later_used_symbols = cur_sir.analyse_outputs(
            runtime_context, user_frames, additional_outputs=outputs_symbols
        )
        cur_sir.outputs = output_symbols

        log(1, "start subgraph compile and execution.\n")
        log(1, self.TOS, "\n")

        # step2: construct inputs
        to_static_inputs = construct_eager_inputs(
            cur_sir.inputs, runtime_context.get_runtime()
        )

        # step3: call compile_sir and get python function, third cache is triggered here.
        static_func = CompileSIRCache()(self, cur_sir.name)

        # step4: execute to_static and get outputs, and clear the name of eager tensor.
        eager_tensor_outputs = static_func(to_static_inputs)
        clear_eager_tensor_name(eager_tensor_outputs)
        log(5, "Input", cur_sir.inputs, to_static_inputs)
        log(
            5,
            static_func.get_concrete_program(to_static_inputs)[1].train_program,
        )
        log(5, "Output", cur_sir.outputs, eager_tensor_outputs)

        # step5: reset runtime_value and proxytensor.
        self.bind_value_to_output_proxy_tensor(
            cur_sir, runtime_context, eager_tensor_outputs
        )

        # step6: GC and reset TOS
        self.reset_TOS()
        return_values = self.fetch_output(output)
        self.gc_pass(runtime_context, later_used_symbols)
        return return_values

    def bind_value_to_output_proxy_tensor(
        self, sir, runtime_context, eager_tensor_outputs
    ):
        assert len(sir.outputs) == len(eager_tensor_outputs)
        for symbol, eager_tensor_output in zip(
            sir.outputs, eager_tensor_outputs
        ):
            runtime_context.runtime_name_to_proxy_tensor[symbol.name].set_value(
                eager_tensor_output
            )

    def gc_pass(self, runtime_context, later_used_symbols):
        for symbol_name in list(
            runtime_context.runtime_name_to_proxy_tensor.keys()
        ):
            if Symbol(symbol_name) not in later_used_symbols:
                runtime_context.clear_proxy_tensor_by_name(symbol_name)

    def compile_do_nothing(self, ret_vals):
        def dummy_func(*args, **kwargs):
            return []

        # return None function
        dummy_stmt_ir = StatementIR("dummy_func")
        dummy_stmt_ir.outputs = []
        dummy_stmt_ir.inputs = []
        return dummy_func, dummy_stmt_ir

    def compile_fn(self, ret_vals):
        """
        start compile and return the python function, which must can be to_static without errors.
        """
        cur_sir: StatementIR = self.TOS
        # step0: if no statement, return a dummy function
        if len(cur_sir.statements) == 0:
            return self.compile_do_nothing(ret_vals)
        # step1: analyse sir inputs and outputs
        cur_sir.inputs = cur_sir.analyse_inputs()
        # TODO: output analysis
        cur_sir.outputs = paddle.utils.map_structure(
            lambda x: Symbol(x.name), ret_vals
        )
        log(1, "start subgraph compile and execution.\n")
        log(1, self.TOS, "\n")
        # step2: call compile_sir and get python function, third cache is triggered here.
        static_func = CompileSIRCache()(self, cur_sir.name)
        # step3: GC and reset TOS
        # self.reset_TOS()

        return static_func, cur_sir

    def fetch_output(self, output):
        return paddle.utils.map_structure(
            lambda x: x.value() if is_proxy_tensor(x) else x, output
        )


def clear_eager_tensor_name(output_tensors):
    for output_tensor in output_tensors:
        output_tensor.name = ""


def construct_eager_inputs(input_names, runtime_value):
    output_list = []
    for inp in input_names:
        assert (
            runtime_value[inp.name].value() is not None
        ), f"Inputs {inp.name} of graph must have value."
        output_list.append(runtime_value[inp.name].value())

    return output_list
