from __future__ import annotations

import operator
import types
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np

import paddle

from ....infer_meta import MetaInfo
from ....symbolic.statement_ir import Symbol
from ....utils import (
    BreakGraphError,
    NameGenerator,
    NotImplementException,
    NotImplementException,
    log_do,
    paddle_tensor_methods,
)
from ....utils.exceptions import InnerError
from ..guard import StringifyExpression, union_free_vars
from ..pycode_generator import PyCodeGen
from ..tracker import (
    ConstTracker,
    DanglingTracker,
    DummyTracker,
    GetAttrTracker,
    Tracker,
)
from .base import ConstTypes, VariableBase, VariableFactory

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph

DTYPE_ABBRS = {
    paddle.bfloat16: 'bfloat16',
    paddle.float64: 'float64',
    paddle.float32: 'float32',
    paddle.float16: 'float16',
    paddle.complex64: 'complex64',
    paddle.complex128: 'complex128',
    paddle.int8: 'int8',
    paddle.int16: 'int16',
    paddle.int32: 'int32',
    paddle.int64: 'int64',
    paddle.bool: 'bool',
    paddle.uint8: 'uint8',
}


class ConstantVariable(VariableBase):
    def __init__(
        self,
        value: Any,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.value = value

    def get_value(self):
        return self.value

    @property
    def debug_name(self) -> str:
        return f"{self.value}"

    @debug_name.setter
    def debug_name(self, name):
        pass

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_const(self.value)

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def __bool__(self) -> bool:
        return bool(self.value)

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, ConstTypes):
            return ConstantVariable(value, tracker)
        return None

    @staticmethod
    def wrap_literal(value: Any) -> ConstantVariable:
        if isinstance(value, ConstantVariable):
            return value
        assert isinstance(
            value, ConstTypes
        ), f"value: {value},type: {type(value)}"
        return ConstantVariable(value, ConstTracker(value))


implemented_property = set()


def tensor_property(func):
    implemented_property.add(func.__name__)
    return property(func)


class TensorVariable(VariableBase):
    var_name_generator = NameGenerator("var_")

    def __init__(
        self,
        tensor: paddle.Tensor | MetaInfo,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        if isinstance(tensor, paddle.Tensor):
            self.value = tensor
            self.meta = MetaInfo.from_tensor(tensor)
        elif isinstance(tensor, MetaInfo):
            self.value = None
            self.meta = tensor
        else:
            raise InnerError(
                "Required type(tensor) is paddle.Tensor or ProxyTensor, but received {}.".format(
                    type(tensor).__name__
                )
            )
        self.var_name = TensorVariable.var_name_generator.next()
        self.graph = graph

    def get_value(self):
        if self.value is None:
            raise InnerError("Can not get value from a inner tensor variable.")
        return self.value

    def get_type(self):
        return paddle.Tensor

    def get_symbol(self) -> Symbol:
        return Symbol(self.var_name)

    @property
    def out_var_name(self):
        return f"{self.graph.out_var_prefix}{self.var_name}"

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_fast(self.out_var_name)

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
            f"str(MetaInfo.from_tensor({frame_value_tracer.expr})) == '{self.meta}'",
            union_free_vars(
                {"MetaInfo": MetaInfo},
                frame_value_tracer.free_vars,
            ),
        )

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "shape": self.meta.shape,
            "dtype": DTYPE_ABBRS[self.meta.dtype],
            "stop_gradient": self.meta.stop_gradient,
            "var_name": self.var_name,
            "var_name": self.var_name,
        }

    def __getitem__(self, key):
        return self.graph.call_tensor_method(
            '__getitem__',
            self,
            VariableFactory.from_value(
                key, self.graph, tracker=ConstTracker(key)
            ),
        )

    def __setitem__(self, key, value):
        return self.graph.call_tensor_method(
            '__setitem__',
            self,
            VariableFactory.from_value(
                key, self.graph, tracker=ConstTracker(key)
            ),
            value,
        )

    @tensor_property
    def T(self):
        perm = list(range(len(self.meta.shape) - 1, -1, -1))
        perm_var = VariableFactory.from_value(
            perm, self.graph, tracker=ConstTracker(perm)
        )
        out = self.graph.call_paddle_api(paddle.transpose, self, perm_var)
        return out

    @tensor_property
    def ndim(self):
        return ConstantVariable.wrap_literal(len(self.meta.shape))

    @tensor_property
    def size(self):
        # TODO: maybe break graph.
        if self.meta.is_dynamic_shape():
            raise BreakGraphError(
                f"Getting size for a dynamic shape tensor causes graph break. shape = {self.meta.shape}"
            )
        elements = reduce(operator.mul, self.meta.shape, 1)
        return ConstantVariable.wrap_literal(elements)

    @tensor_property
    def shape(self):
        if self.meta.is_dynamic_shape():
            raise BreakGraphError(
                f"Getting shape for a dynamic shape tensor causes graph break. shape = {self.meta.shape}"
            )
        self.graph.add_global_guarded_variable(self)
        return VariableFactory.from_value(
            self.meta.shape, self.graph, tracker=ConstTracker(self.meta.shape)
        )

    def is_tensor(self):
        return ConstantVariable.wrap_literal(True)

    def is_complex(self):
        dtype = self.meta.dtype
        is_cp_dtype = dtype == paddle.complex64 or dtype == paddle.complex128
        return ConstantVariable.wrap_literal(is_cp_dtype)

    def is_integer(self):
        dtype = self.meta.dtype
        is_int_dtype = (
            dtype == paddle.int8
            or dtype == paddle.uint8
            or dtype == paddle.int16
            or dtype == paddle.int32
            or dtype == paddle.int64
        )
        return ConstantVariable.wrap_literal(is_int_dtype)

    def is_floating_point(self):
        dtype = self.meta.dtype
        is_fp_dtype = (
            dtype == paddle.float32
            or dtype == paddle.float64
            or dtype == paddle.float16
            or dtype == paddle.bfloat16
        )
        return ConstantVariable.wrap_literal(is_fp_dtype)

    def getattr(self, name: str):
        method_name_to_builtin_fn = {
            "dim": paddle.rank,
            "ndimension": paddle.rank,
            "is_tensor": paddle.is_tensor,
            "is_complex": paddle.is_complex,
            "is_integer": paddle.is_integer,
            "is_floating_point": paddle.is_floating_point,
        }
        if name in ["dtype", "type", "persistable", "name", "stop_gradient"]:
            return VariableFactory.from_value(
                getattr(self.meta, name),
                self.graph,
                tracker=GetAttrTracker(self, name),
            )
        elif name in implemented_property:
            return getattr(self, name)
        elif name in method_name_to_builtin_fn:
            # TODO: backward, gradient
            from .callable import BuiltinVariable

            builtin_fn = method_name_to_builtin_fn[name]

            return BuiltinVariable(
                builtin_fn, self.graph, DanglingTracker()
            ).bind(self, name)
        elif name in paddle_tensor_methods:
            from .callable import TensorFunctionVariable

            fn_var = TensorFunctionVariable(
                name, graph=self.graph, tracker=DanglingTracker()
            )
            return fn_var.bind(self, name)
        else:
            raise InnerError(f"Unknown Tensor attribute: {name}")

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, (paddle.Tensor, MetaInfo)):
            assert graph is not None
            return TensorVariable(value, graph, tracker)
        return None


class ObjectVariable(VariableBase):
    def __init__(self, obj, graph, tracker):
        super().__init__(tracker)
        self.value = obj
        self.graph = graph

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def get_value(self) -> Any:
        return self.value


class SliceVariable(VariableBase):
    def __init__(self, slice_: slice, graph, tracker):
        super().__init__(tracker)
        self.value = slice_
        self.graph = graph

    @property
    def debug_name(self) -> str:
        return ":".join(
            [
                str(self.value.start) if self.value.start is not None else "",
                str(self.value.stop) if self.value.stop is not None else "",
                str(self.value.step) if self.value.step is not None else "",
            ]
        )

    @debug_name.setter
    def debug_name(self, name):
        pass

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def get_value(self):
        return self.value

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, slice):
            return SliceVariable(value, graph, tracker)
        return None


class ModuleVariable(VariableBase):
    def __init__(self, func, graph, tracker):
        super().__init__(tracker)
        self.value = func
        self.graph = graph

    def get_value(self):
        return self.value

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, types.ModuleType):
            return ModuleVariable(value, graph, tracker)
        return None


class DygraphTracerVariable(VariableBase):
    # TODO(SigureMo): Remove this trick after we add CompareTracker
    def __init__(self, value, graph, tracker):
        super().__init__(tracker)
        self.value = value
        self.graph = graph

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
        return StringifyExpression("True", {})

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "is_none": self.value is None,
        }

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, paddle.fluid.dygraph.tracer.Tracer):
            return DygraphTracerVariable(value, graph, tracker)
        return None


class NumpyVariable(VariableBase):
    def __init__(self, value, graph, tracker):
        super().__init__(tracker)
        self.value = value
        self.graph = graph

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def get_value(self) -> Any:
        return self.value

    def make_stringify_guard(self) -> StringifyExpression:
        raise NotImplementException("We can not stringify numpy variable")

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, (np.ndarray, np.number)):
            return NumpyVariable(value, graph, tracker)
        return None


class DummyVariable(VariableBase):
    def __init__(self):
        super().__init__(DanglingTracker())

    def reconstruct(self, codegen: PyCodeGen):
        codegen.gen_push_null()
