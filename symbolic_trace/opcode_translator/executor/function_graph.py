# This file is specifically used to handle the problem
# of generating a Graph from a linear function call.

from __future__ import annotations

import paddle

from ...infer_meta import InferMetaCache, infer_meta
from ...proxy_tensor import ProxyTensor, ProxyTensorContext
from ...symbolic.statement_ir import Symbol
from ...symbolic.symbolic_context import SymbolicTraceContext
from ...utils import is_paddle_api, log
from .pycode_generator import PyCodeGen
from .tracker import DummyTracker
from .variables import (
    ContainerVariable,
    Guard,
    TensorVariable,
    VariableTracker,
    VariableTrackerFactory,
    compose_guards,
    topo_sort_vars,
)


def convert_to_meta(inputs):
    def func(x):
        if isinstance(x, ProxyTensor):
            return x.meta
        return x

    return paddle.utils.map_structure(func, inputs)


def convert_to_symbol(inputs):
    def func(x):
        if isinstance(x, ProxyTensor):
            return Symbol(x.name)
        return x

    pack_inputs = inputs
    if not paddle.utils.is_sequence(inputs):
        pack_inputs = [inputs]
    ret = paddle.utils.map_structure(func, pack_inputs)
    if not paddle.utils.is_sequence(inputs):
        ret = ret[0]
    return ret


def convert_variable_to_value(inputs):
    def func(x):
        return x.get_value()

    return paddle.utils.map_structure(func, inputs)


class FunctionGraph:
    """
    A Graph representation corresponding to each FunctionFrame
    The input binding diagram containing the current call represents three parts of output settings,
    This Graph can be compiled as a f_locals dependency function which produce the same outputs.
    """

    def __init__(self, frame):
        self.sir_ctx = SymbolicTraceContext()
        self.inner_out = set()
        self.input_variables = []
        self.pycode_gen = PyCodeGen(frame)
        self.py_frame = frame
        self.out_var_prefix = "___SIR_out_"
        self._global_guarded_variables: list[VariableTracker] = []

    def need_add_input(self, var):
        if var.id in self.inner_out:
            return False
        for v in self.input_variables:
            if v.id == var.id:
                return False
        return True

    def collect_input_variables(self, inputs: list[VariableTracker]):
        for inp in inputs:
            if isinstance(inp, ContainerVariable):
                self.collect_input_variables(inp.get_items())
            if isinstance(inp, VariableTracker) and self.need_add_input(inp):
                self.input_variables.append(inp)

    @property
    def guard_fn(self) -> Guard:
        guards = [
            variable.make_check_fn()
            for variable in topo_sort_vars(
                self.input_variables + self._global_guarded_variables
            )
            if not isinstance(variable.tracker, DummyTracker)
        ]
        for guard in guards:
            assert callable(guard), "guard must be callable."

        return compose_guards(guards)

    def start_compile(self, *ret_vars: VariableTracker):
        ret_items = [
            ret_item
            for ret_var in ret_vars
            for ret_item in ret_var.flatten_items()
        ]
        tensor_items = self._find_tensor_outputs(ret_items)
        compiled_fn, statment_ir = self.sir_ctx.compile_fn(
            [tensor_var.value for tensor_var in tensor_items]
        )
        input_names = statment_ir.inputs
        compiled_fn_name = statment_ir.name
        # prepare function and inputs
        self.pycode_gen.gen_load_object(compiled_fn, compiled_fn_name)
        for name in input_names:
            found = False
            for variable in self.input_variables:
                if (
                    isinstance(variable, TensorVariable)
                    and variable.value.name == name
                ):
                    variable.tracker.gen_instructions(self.pycode_gen)
                    found = True
                    break
            assert found, f"can't find input {name} in SIR."
        # Pack all args into a tuple, because we don't support *args now.
        self.pycode_gen.gen_build_tuple(count=len(input_names))
        # call the compiled_fn
        self.pycode_gen.gen_call_function(argc=1)
        # Store outputs to f_locals
        self.pycode_gen.gen_unpack_sequence(count=len(tensor_items))
        for tensor_var in tensor_items:
            self.pycode_gen.gen_store_fast(tensor_var.out_var_name)
        # restore the outputs.
        for ret_var in ret_vars:
            ret_var.reconstruct(self.pycode_gen)

        # deal side effect
        # TODO(xiongkun): add side effect handle

    def call_paddle_api(self, func, *args, **kwargs):
        """
        Inputs is a lots of VariableTracker.
        """
        assert is_paddle_api(func)
        # not fallback api, start symbolic trace.
        # TODO(xiokgun): multi-output support.
        # TODO(xiokgun): may have python buildin object inside metas.
        # TODO(xiokgun): 4 kinds of python arguments. support it !!
        log(3, f"call paddle.api : {func.__name__}", "\n")
        self.collect_input_variables(list(args))
        self.collect_input_variables(list(kwargs.values()))
        values, kwvalues = (
            convert_variable_to_value(args),
            convert_variable_to_value(kwargs),
        )
        metas = convert_to_meta(values)
        kwmetas = convert_to_meta(kwvalues)
        meta = InferMetaCache()(func, *metas, **kwmetas)
        result = ProxyTensor(ProxyTensorContext().new_varname(), meta)
        inputs_symbols = (
            convert_to_symbol(values),
            convert_to_symbol(kwvalues),
        )
        log(3, f"         inputs : {inputs_symbols}", "\n")
        self.sir_ctx.call_API(
            func,
            inputs=inputs_symbols,
            outputs=convert_to_symbol(result),
        )  # symbolic only contain symbols.
        variable = VariableTrackerFactory.from_value(
            result,
            self,
            tracker=DummyTracker(list(args) + list(kwargs.values())),
        )
        self._put_inner(variable)
        return variable

    def call_tensor_method(self, method_name, *args):
        """
        Inputs is a lots of VariableTracker.
        """
        self.collect_input_variables(list(args))
        values = convert_variable_to_value(args)
        metas = convert_to_meta(values)
        meta = infer_meta(method_name, *metas)
        result = ProxyTensor(ProxyTensorContext().new_varname(), meta)
        self.sir_ctx.call_METHOD(
            method_name,
            inputs=(convert_to_symbol(values), {}),
            outputs=convert_to_symbol(result),
        )  # symbolic only contain symbols.
        variable = VariableTrackerFactory.from_value(
            result, self, tracker=DummyTracker(list(args))
        )
        self._put_inner(variable)
        return variable

    def _put_inner(self, var):
        self.inner_out.add(var.id)

    def add_global_guarded_variable(self, variable: VariableTracker):
        self._global_guarded_variables.append(variable)

    def _find_tensor_outputs(
        self, outputs: list[VariableTracker]
    ) -> list[TensorVariable]:
        output_tensors: list[TensorVariable] = []
        for output in outputs:
            if isinstance(output, TensorVariable) and isinstance(
                output.tracker, DummyTracker
            ):
                output_tensors.append(output)
        return output_tensors
