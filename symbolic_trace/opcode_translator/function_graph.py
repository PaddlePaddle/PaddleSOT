# This file is specifically used to handle the problem
# of generating a Graph from a linear function call.

import paddle

from ..infer_meta import InferMetaCache, MetaInfo, infer_meta
from ..proxy_tensor import ProxyTensor, ProxyTensorContext, callable_wrapper
from ..symbolic.statement_ir import Symbol
from ..symbolic.symbolic_context import SymbolicTraceContext
from .pycode_generator import PyCodeGen
from .variables import TensorVariable, VariableTracker, VariableTrackerFactory


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


class FunctionGraph:
    """
    A Graph representation corresponding to each FunctionFrame
    The input binding diagram containing the current call represents three parts of output settings,
    This Graph can be compiled as a f_locals dependency function which produce the same outputs.
    """

    def __init__(self, frame):
        self.sir_ctx = SymbolicTraceContext()
        self.inner_out = set()
        self.input_trackers = []
        self.pycode_gen = PyCodeGen(frame)
        self.py_frame = frame

    def collect_input_trackers(self, inputs):
        outputs = []
        for inp in inputs:
            if isinstance(inp, VariableTracker):
                if inp.id not in self.inner_out and inp.source is not None:
                    self.input_trackers.append(inp)
                outputs.append(inp.value)
        return outputs

    @property
    def guard_fn(self):
        guards = [tracker.make_check_fn() for tracker in self.input_trackers]

        def _guard_fn(frame):
            ret = True
            for guard in guards:
                ret = ret and guard(frame)
            return ret

        return _guard_fn

    def start_compile(self, ret_val):
        assert isinstance(ret_val, TensorVariable), "Not Implement yet."
        compiled_fn, statment_ir = self.sir_ctx.compile_fn(ret_val.value)
        input_names = statment_ir.inputs
        compiled_fn_name = statment_ir.name
        # prepare function and inputs
        self.pycode_gen.gen_load_object(compiled_fn, compiled_fn_name)
        for name in input_names:
            for tracker in self.input_trackers:
                if (
                    isinstance(tracker, TensorVariable)
                    and tracker.value.name == name
                ):
                    self.pycode_gen.add_pure_instructions(
                        tracker.source.gen_instructions()
                    )
        # call the compiled_fn
        self.pycode_gen.gen_call_function(argc=1)
        # restore the outputs.
        # TODO(xiongkun): add side effect handle

        # return
        self.pycode_gen.gen_return()
        new_code = self.pycode_gen.gen_pycode()
        return new_code, self.guard_fn

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
        args, kwargs = self.collect_input_trackers(
            args
        ), self.collect_input_trackers(kwargs)
        metas = convert_to_meta(args)
        kwmetas = convert_to_meta(kwargs)
        meta = InferMetaCache()(func, *metas, **kwmetas)
        result = ProxyTensor(SymbolicTraceContext().new_varname(), meta)
        inputs_symbols = (convert_to_symbol(args), convert_to_symbol(kwargs))
        log(3, f"         inputs : {inputs_symbols}", "\n")
        SymbolicTraceContext().call_API(
            func, inputs=inputs_symbols, outputs=convert_to_symbol(result)
        )  # symbolic only contain symbols.
        variable = VariableTrackerFactory.from_value(result, self)
        self._put_inner(variable)
        return variable

    def call_tensor_method(self, method_name, *args):
        """
        Inputs is a lots of VariableTracker.
        """
        args = self.collect_input_trackers(args)
        metas = convert_to_meta(args)
        meta = infer_meta(method_name, *metas)
        result = ProxyTensor(SymbolicTraceContext().new_varname(), meta)
        SymbolicTraceContext().call_METHOD(
            method_name,
            inputs=(convert_to_symbol(args), {}),
            outputs=convert_to_symbol(result),
        )  # symbolic only contain symbols.
        variable = VariableTrackerFactory.from_value(result, self)
        self._put_inner(variable)
        return variable

    def _put_inner(self, var):
        self.inner_out.add(var.id)
