from __future__ import annotations

import inspect
import types
from typing import TYPE_CHECKING, Any, Callable

import paddle

from ....symbolic.statement_ir import Symbol
from ....utils import (
    ASSERT,
    NameGenerator,
    is_break_graph_api,
    is_break_graph_tensor_methods,
    is_paddle_api,
    log_do,
)
from ....utils.exceptions import BreakGraphError, FallbackErrorBase
from ..guard import StringifyExpression, union_free_vars
from ..tracker import DummyTracker, GetAttrTracker, GetItemTracker, Tracker
from .base import VariableBase, VariableFactory
from .basic import ConstantVariable, TensorVariable

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph


class CallableVariable(VariableBase):
    def __init__(self, graph: FunctionGraph, tracker: Tracker):
        super().__init__(tracker)
        self.graph = graph

    def __call__(self, *args, **kwargs) -> VariableBase:
        return self.call_function(*args, **kwargs)

    def call_function(self, *args, **kwargs):
        raise NotImplementedError("call_function is not implemented.")


class FunctionVariable(CallableVariable):
    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(graph, tracker)
        self.value = fn

    def get_value(self):
        return self.value

    def get_code(self) -> types.CodeType:
        return self.value.__code__


class UserDefinedFunctionVariable(FunctionVariable):
    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(fn, graph, tracker)

    def call_function(self, *args, **kwargs) -> VariableBase:
        from ..opcode_inline_executor import OpcodeInlineExecutor

        if self.value is ASSERT:
            return self.value(args[0].value)

        checkpoint = self.graph.save_memo()
        try:
            inline_executor = OpcodeInlineExecutor(self, *args, **kwargs)
            output = inline_executor.inline_call()
        except FallbackErrorBase as e:
            self.graph.restore_memo(checkpoint)
            raise BreakGraphError(
                f"{self.value} is raise a inline call error. {e}"
            )
        return output

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, (types.FunctionType)):
            return UserDefinedFunctionVariable(value, graph, tracker)
        return None

    def __repr__(self) -> str:
        return f"UserDefinedFunctionVariable({self.value.__name__})"


class PaddleApiVariable(FunctionVariable):
    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(fn, graph, tracker)

    def call_function(self, *args, **kwargs):
        if is_break_graph_api(self.value):
            raise BreakGraphError()
        return self.graph.call_paddle_api(self.value, *args, **kwargs)

    @VariableFactory.register_from_value(
        successor="UserDefinedFunctionVariable"
    )
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if callable(value) and is_paddle_api(value):
            return PaddleApiVariable(value, graph, tracker)
        return None

    def __repr__(self) -> str:
        return f"PaddleApiVariable({self.value.__name__})"


class MethodVariable(CallableVariable):
    def __init__(
        self,
        bound_instance: VariableBase,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.bound_instance = bound_instance


class UserDefinedMethodVariable(MethodVariable):
    def __init__(
        self, bound_instance, fn, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(bound_instance, graph, tracker)
        self.bound_instance = bound_instance
        self.fn = fn

    def get_value(self):
        return self.fn.__get__(
            self.bound_instance, self.bound_instance.__class__
        )

    def call_function(self, *args, **kwargs):
        fn_var = UserDefinedFunctionVariable(
            self.fn, self.graph, GetAttrTracker(self, "__func__")
        )

        return fn_var(*(self.bound_instance, *args), **kwargs)

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if inspect.ismethod(value):
            method_self = VariableFactory.from_value(
                value.__self__, graph, DummyTracker([])
            )
            method_var = UserDefinedMethodVariable(
                method_self,
                value.__func__,
                graph,
                tracker,
            )
            method_self.tracker = GetAttrTracker(method_var, "__self__")
            return method_var
        return None

    def __repr__(self) -> str:
        return f"UserDefinedMethodVariable({self.fn.__name__})"


class TensorMethodVariable(MethodVariable):
    def __init__(
        self,
        tensor: TensorVariable,
        method_name: str,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tensor, graph, tracker)
        self.tensor = tensor
        self.method_name = method_name

    def get_value(self):
        return getattr(self.tensor, self.method_name)

    def call_function(self, *args, **kwargs):
        if is_break_graph_tensor_methods(self.method_name):
            raise BreakGraphError(
                f"Break graph by tensor methods: {self.method_name}"
            )
        return self.graph.call_tensor_method(
            self.method_name, self.tensor, *args, **kwargs
        )

    def _reconstruct(self, pycode_gen):
        self.tensor.reconstruct(pycode_gen)
        pycode_gen.gen_load_attr(self.method_name)

    @VariableFactory.register_from_value(successor="UserDefinedMethodVariable")
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if inspect.ismethod(value) and isinstance(
            value.__self__, paddle.Tensor
        ):
            # NOTE(SigureMo): Since the method_self need method_var as the obj
            # of the tracker, we need to temporarily set the tracker of method_self
            # to DummyTracker, and set it to GetAttrTracker after method_var is created.
            method_self = TensorVariable(
                value.__self__, graph, DummyTracker([])
            )
            method_var = TensorMethodVariable(
                method_self,
                value.__name__,
                graph,
                tracker,
            )
            method_self.tracker = GetAttrTracker(method_var, "__self__")
            return method_var
        return None

    def __repr__(self) -> str:
        return f"TensorMethodVariable({self.method_name})"


class DirectlyCallMethodVariable(MethodVariable):
    def __init__(
        self, bound_instance, fn, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(bound_instance, graph, tracker)
        self.bound_instance = bound_instance
        self.fn = fn

    def get_value(self):
        return self.fn.__get__(
            self.bound_instance, self.bound_instance.__class__
        )

    def call_function(self, *args, **kwargs):
        return self.fn(*(self.bound_instance, *args), **kwargs)


class LayerVariable(CallableVariable):
    def __init__(
        self, layer: paddle.nn.Layer, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(graph, tracker)
        self.value = layer

    def get_value(self):
        return self.value

    def make_stringify_guard(self) -> StringifyExpression:
        assert not isinstance(
            self.tracker, DummyTracker
        ), "Can not make guard from dummy tracker"

        frame_value_tracer = self.tracker.trace_value_from_frame()
        log_do(
            4,
            lambda: print(
                f"[Guard]: guard_fn for {self}, tracker={self.tracker.__class__.__name__}, value={frame_value_tracer.expr}"
            ),
        )
        return StringifyExpression(
            f"id({frame_value_tracer.expr}) == {id(self.get_value())}",
            union_free_vars(frame_value_tracer.free_vars),
        ) & StringifyExpression(
            f"{frame_value_tracer.expr}.training == {self.get_value().training}",
            union_free_vars(frame_value_tracer.free_vars),
        )


class UserDefinedLayerVariable(LayerVariable):
    def __init__(
        self, layer: paddle.nn.Layer, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(layer, graph, tracker)

    def call_function(self, *args, **kwargs):
        fn_var = UserDefinedFunctionVariable(
            self.value.__class__.__call__,
            self.graph,
            GetAttrTracker(self, "__call__"),
        )

        return fn_var(*(self, *args), **kwargs)

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(
            value, paddle.nn.Layer
        ) and not value.__module__.startswith("paddle.nn."):
            return UserDefinedLayerVariable(value, graph, tracker)
        return None

    def __repr__(self) -> str:
        return f"UserDefinedLayerVariable({self.value.__class__.__name__})"


class BuiltinVariable(CallableVariable):
    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(graph, tracker)
        self.value = fn

    def call_function(self, *args, **kwargs):
        # TODO(0x45f): For builtin functions, may have 3 different ways to process as below:
        #     1. Simulation execution: ensure correct simulation execution and handle trackers with care
        #     2. Trigger the paddle api call
        #     3. Trigger fallback
        if is_break_graph_api(self.value):
            raise BreakGraphError()
        args = [
            arg.value if isinstance(arg, ConstantVariable) else arg
            for arg in args
        ]
        kwargs = {
            k: (v.value if isinstance(v, ConstantVariable) else v)
            for k, v in kwargs.items()
        }
        return self.value(*args, **kwargs)

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, (types.BuiltinFunctionType)):
            return BuiltinVariable(value, graph, tracker)
        return None

    def __repr__(self) -> str:
        return f"BuiltinVariable({self.value.__name__})"


class UserDefinedGeneratorVariable(FunctionVariable):
    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(fn, graph, tracker)

    def call_function(self, *args, **kwargs) -> VariableBase:

        iter_ = self.value()
        return VariableFactory.from_value(
            iter_, self.graph, DummyTracker([self])
        )

    @VariableFactory.register_from_value(
        successor="UserDefinedFunctionVariable"
    )
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if inspect.isgeneratorfunction(value):
            return UserDefinedGeneratorVariable(value, graph, tracker)
        return None

    def __repr__(self) -> str:
        return f"UserDefinedGeneratorVariable({self.value.__name__})"


class PaddleLayerVariable(LayerVariable):
    layer_name_generator = NameGenerator("layer_")

    def __init__(
        self, layer: paddle.nn.Layer, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(layer, graph, tracker)
        self.name = self.layer_name_generator.next()

    def get_symbol(self) -> Symbol:
        return Symbol(self.name)

    def call_function(self, *args, **kwargs):
        # TODO: Remove this trick after we support for-loop.
        if isinstance(self.value, paddle.nn.Sequential):
            assert len(args) == 1, "Sequential only accept one input"
            input = args[0]
            for i, layer in enumerate(self.value._sub_layers.values()):
                layer_var = VariableFactory.from_value(
                    layer, self.graph, tracker=GetItemTracker(self, i)
                )
                assert isinstance(layer_var, LayerVariable)
                input = layer_var(input)
            return input
        return self.graph.call_layer(self, *args, **kwargs)

    @VariableFactory.register_from_value(successor="UserDefinedLayerVariable")
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        # TODO(SigureMo): Add a more common way to check if a value is a paddle builtin layer.
        if isinstance(value, paddle.nn.Layer) and value.__module__.startswith(
            "paddle.nn."
        ):
            return PaddleLayerVariable(value, graph, tracker)
        return None

    def __repr__(self) -> str:
        return f"PaddleLayerVariable({self.value.__class__.__name__})"
