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
    is_builtin_fn,
    is_paddle_api,
    log_do,
    magic_method_builtin_dispatch,
    psdb_print,
)
from ....utils.exceptions import BreakGraphError, FallbackErrorBase
from ..dispatcher import Dispatcher
from ..guard import StringifyExpression, union_free_vars
from ..tracker import (
    DanglingTracker,
    DummyTracker,
    GetAttrTracker,
    GetItemTracker,
    Tracker,
)
from .base import VariableBase, VariableFactory
from .basic import ConstantVariable, PrintStmtVariable

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph


class CallableVariable(VariableBase):
    def __init__(self, graph: FunctionGraph, tracker: Tracker):
        super().__init__(graph, tracker)

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

    def bind(self, instance: VariableBase, name: str):
        method_var = MethodVariable(
            instance,
            self,
            graph=self.graph,
            tracker=GetAttrTracker(instance, name),
        )
        class_var = VariableFactory.from_value(
            instance.get_type(),
            graph=self.graph,
            tracker=GetAttrTracker(instance, "__class__"),
        )
        assert class_var is not None
        self.tracker = GetAttrTracker(class_var, name)
        return method_var


class UserDefinedFunctionVariable(FunctionVariable):
    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(fn, graph, tracker)

    def call_function(self, *args, **kwargs) -> VariableBase:
        from ..opcode_inline_executor import OpcodeInlineExecutor

        # special function for inner debug.
        if self.value is ASSERT:
            # TODO: add comptime check mechanism
            return ConstantVariable.wrap_literal(
                self.value(args[0].value), self.graph
            )
        if self.value is psdb_print:
            sot_prefix = ConstantVariable.wrap_literal("[SOT]", self.graph)
            self.graph.add_print_variables(
                PrintStmtVariable(([sot_prefix, *args], kwargs), self.graph)
            )
            return ConstantVariable.wrap_literal(None, self.graph)

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
        if isinstance(value, (types.FunctionType)) and graph is not None:
            return UserDefinedFunctionVariable(value, graph, tracker)
        return None

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "name": self.value.__name__,
        }


class PaddleApiVariable(FunctionVariable):
    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(fn, graph, tracker)

    def call_function(self, *args, **kwargs):
        if is_break_graph_api(self.value):
            raise BreakGraphError(
                f"breakgraph by unsupport function: {self.value.__name__}"
            )
        return self.graph.call_paddle_api(self.value, *args, **kwargs)

    @VariableFactory.register_from_value(
        successor="UserDefinedFunctionVariable"
    )
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if callable(value) and is_paddle_api(value) and graph is not None:
            return PaddleApiVariable(value, graph, tracker)
        return None

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "name": self.value.__name__,
        }


class TensorFunctionVariable(FunctionVariable):
    def __init__(
        self, method_name: str, graph: FunctionGraph, tracker: Tracker
    ):
        fn = getattr(paddle.static.Variable, method_name)
        super().__init__(fn, graph, tracker)
        self.method_name = method_name

    def call_function(self, *args, **kwargs):
        if is_break_graph_tensor_methods(self.method_name):
            raise BreakGraphError()
        return self.graph.call_tensor_method(self.method_name, *args, **kwargs)

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "name": self.value.__name__,
        }


class MethodVariable(CallableVariable):
    def __init__(
        self,
        bound_instance: VariableBase,
        fn: VariableBase,
        graph: FunctionGraph,
        tracker: Tracker,
        *,
        method_name: str | None = None,
    ):
        super().__init__(graph, tracker)
        self.bound_instance = bound_instance
        self.fn = fn
        self.method_name = method_name

    def get_value(self):
        return self.fn.get_value().__get__(
            self.bound_instance.get_value(),
            self.bound_instance.get_value().__class__,
        )

    def _reconstruct(self, pycode_gen):
        self.graph.add_global_guarded_variable(self)
        assert self.method_name is not None
        self.tensor.reconstruct(pycode_gen)
        pycode_gen.gen_load_attr(self.method_name)

    def call_function(self, *args, **kwargs):
        return self.fn(*(self.bound_instance, *args), **kwargs)

    @staticmethod
    def wrap_method(
        value: types.MethodType,
        *,
        tracker: Tracker,
        instance: VariableBase | None = None,
        fn: VariableBase | None = None,
        method_name: str | None = None,
        graph: FunctionGraph | None = None,
    ):
        # NOTE(SigureMo): Since the method_self need method_var as the obj
        # of the tracker, we need to temporarily set the tracker of method_self
        # to DummyTracker, and set it to GetAttrTracker after method_var is created.
        if instance is None:
            instance_var = VariableFactory.from_value(
                value.__self__, graph, DanglingTracker()
            )
        else:
            instance_var = instance

        if fn is None:
            fn_var = VariableFactory.from_value(
                value.__func__, graph, DanglingTracker()
            )
        else:
            fn_var = fn

        assert graph is not None
        method_var = MethodVariable(
            instance_var,
            fn_var,
            method_name=method_name,
            graph=graph,
            tracker=tracker,
        )
        if instance is None:
            instance_var.tracker = GetAttrTracker(method_var, "__self__")
        if fn is None:
            fn_var.tracker = GetAttrTracker(method_var, "__func__")
        return method_var

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if inspect.ismethod(value):
            return MethodVariable.wrap_method(
                value=value, tracker=tracker, graph=graph
            )
        return None

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
        }


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

    @VariableFactory.register_from_value(successor="PaddleApiVariable")
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if (
            isinstance(value, paddle.nn.Layer)
            and not value.__module__.startswith("paddle.nn.")
            and graph is not None
        ):
            return UserDefinedLayerVariable(value, graph, tracker)
        return None

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "name": self.value.__class__.__name__,
        }


class BuiltinVariable(FunctionVariable):
    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(fn, graph, tracker)
        self.value = fn

    def call_function(self, *args, **kwargs):
        # Lookup the handler from dispatcher
        handler = Dispatcher.dispatch(self.value, *args, **kwargs)
        if handler is not None:
            return handler(*args, **kwargs)

        # Try to inline call the magic function
        magic_methods = magic_method_builtin_dispatch(self.value)
        for magic_method in magic_methods:
            sorted_args = args
            if magic_method.is_reverse:
                sorted_args = sorted_args[::-1]
            arg_type = sorted_args[0].get_type()
            if hasattr(arg_type, magic_method.name):
                class_fn = getattr(arg_type, magic_method.name)
                class_var = VariableFactory.from_value(
                    arg_type,
                    self.graph,
                    GetAttrTracker(args[0], "__class__"),
                )
                assert isinstance(class_var, VariableBase)
                fn_var = VariableFactory.from_value(
                    class_fn,
                    self.graph,
                    GetAttrTracker(class_var, class_fn.__name__),
                )
                assert isinstance(fn_var, VariableBase)
                return fn_var(*args)

        # Break graph if neither of the above conditions is met
        raise BreakGraphError(
            f"Not support builtin function: {self.value.__name__}"
        )

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if is_builtin_fn(value) and graph is not None:
            return BuiltinVariable(value, graph, tracker)
        return None

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "name": self.value.__name__,
        }


class UserDefinedGeneratorVariable(FunctionVariable):
    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(fn, graph, tracker)

    def call_function(self, *args, **kwargs) -> VariableBase:
        iter_ = self.value()
        var = VariableFactory.from_value(
            iter_, self.graph, DummyTracker([self])
        )
        return var

    @VariableFactory.register_from_value(
        successor="UserDefinedFunctionVariable"
    )
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if inspect.isgeneratorfunction(value) and graph is not None:
            return UserDefinedGeneratorVariable(value, graph, tracker)
        return None

    @property
    def main_info(self) -> dict[str, Any]:
        return {"name": self.value.__name__}


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
        if (
            isinstance(value, paddle.nn.Layer)
            and value.__module__.startswith("paddle.nn.")
            and graph is not None
        ):
            return PaddleLayerVariable(value, graph, tracker)
        return None

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "name": self.value.__class__.__name__,
        }
