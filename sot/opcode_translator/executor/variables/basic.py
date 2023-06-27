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


FP_DTYPE_ABBRS = {
    paddle.bfloat16: 'bfloat16',
    paddle.float64: 'float64',
    paddle.float32: 'float32',
    paddle.float16: 'float16',
}

CP_DTYPE_ABBRS = {
    paddle.complex64: 'complex64',
    paddle.complex128: 'complex128',
}

INT_DTYPE_ABBRS = {
    paddle.int8: 'int8',
    paddle.int16: 'int16',
    paddle.int32: 'int32',
    paddle.int64: 'int64',
    paddle.uint8: 'uint8',
}

DTYPE_ABBRS = {
    **FP_DTYPE_ABBRS,
    **CP_DTYPE_ABBRS,
    **INT_DTYPE_ABBRS,
    paddle.bool: 'bool',
}


class ConstantVariable(VariableBase):
    """
    ConstantVariable is a subclass of VariableBase used to wrap a Variable of the const type.

    Args:
        value(Any): The value to be wrapped.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

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

    def bool(self):
        return VariableFactory.from_value(
            bool(self), self.graph, DummyTracker([self])
        )

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, ConstTypes):
            return ConstantVariable(value, tracker)
        return None

    @staticmethod
    def wrap_literal(value: Any) -> ConstantVariable:
        """
        Wrap a literal value in a ConstantVariable.

        Args:
            value(Any): The literal value to be wrapped.

        Returns:
            ConstantVariable: A new ConstantVariable object that wraps the given value.
        """
        if isinstance(value, ConstantVariable):
            return value
        assert isinstance(
            value, ConstTypes
        ), f"value: {value},type: {type(value)}"
        return ConstantVariable(value, ConstTracker(value))


IMPLEMENTED_TENSOR_PROPERTIES = set()


def tensor_property(func):
    IMPLEMENTED_TENSOR_PROPERTIES.add(func.__name__)
    return property(func)


class TensorVariable(VariableBase):
    """
    TensorVariable is a subclass of VariableBase used to wrap a Variable of the tensor type.

    Args:
        tensor (paddle.Tensor | MetaInfo): The tensor to be wrapped.
        graph (FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker (Tracker): The Tracker object that tracks the information of this variable.
    """

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
            f"MetaInfo.from_tensor({frame_value_tracer.expr}).guard_str() == '{self.meta.guard_str()}'",
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
        }

    def getitem(self, key):
        return self.graph.call_tensor_method(
            '__getitem__',
            self,
            VariableFactory.from_value(
                key, self.graph, tracker=ConstTracker(key)
            ),
        )

    def setitem(self, key, value):
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
        """
        Return a new TensorVariable object that wraps the result of calling the transpose method on the wrapped value of this TensorVariable.
        """
        perm = list(range(len(self.meta.shape) - 1, -1, -1))
        perm_var = VariableFactory.from_value(
            perm, self.graph, tracker=ConstTracker(perm)
        )
        out = self.graph.call_paddle_api(paddle.transpose, self, perm_var)
        return out

    @tensor_property
    def ndim(self):
        """
        Return a ConstantVariable object that represents the number of dimensions of the wrapped value of this TensorVariable.
        """
        return ConstantVariable.wrap_literal(len(self.meta.shape))

    @tensor_property
    def size(self):
        """
        Return a ConstantVariable object that represents the total number of elements in the wrapped value of this TensorVariable.
        """
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
        is_cp_dtype = dtype in CP_DTYPE_ABBRS
        return ConstantVariable.wrap_literal(is_cp_dtype)

    def is_integer(self):
        dtype = self.meta.dtype
        is_int_dtype = dtype in INT_DTYPE_ABBRS
        return ConstantVariable.wrap_literal(is_int_dtype)

    def is_floating_point(self):
        dtype = self.meta.dtype
        is_fp_dtype = dtype in FP_DTYPE_ABBRS
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
        if name in ["dtype", "type", "name", "persistable", "stop_gradient"]:
            if name == "name" and self.meta.name.startswith(
                "infer_meta_variable_tmp"
            ):
                raise BreakGraphError(f"{self.meta.name} is a middle tensor.")
            return VariableFactory.from_value(
                getattr(self.meta, name),
                self.graph,
                tracker=GetAttrTracker(self, name),
            )
        elif name in IMPLEMENTED_TENSOR_PROPERTIES:
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
    """
    ObjectVariable is a subclass of VariableBase used to wrap a Variable of the object type.

    Args:
        obj(Any): The object to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

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
    """
    SliceVariable is a subclass of VariableBase used to wrap a Variable of the slice type.

    Args:
        slice_(slice): The slice to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

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
    """
    ModuleVariable is a subclass of VariableBase used to wrap a Variable of the module type.

    Args:
        func: The module to be wrapped.
        graph: The FunctionGraph object that this variable is associated with.
        tracker: The Tracker object that tracks the information of this variable.
    """

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
    """
    NumpyVariable is a subclass of VariableBase used to wrap a Variable of the numpy type.

    Args:
        value: The numpy value to be wrapped.
        graph: The FunctionGraph object that this variable is associated with.
        tracker: The Tracker object that tracks the information of this variable.
    """

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
        if isinstance(self.get_value(), np.number):
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

            def format_dtype(dtype: np.dtype):
                return f"np.{str(dtype)}"

            def format_number(number: np.number):
                return f"{format_dtype(number.dtype)}({str(number.item())})"

            return StringifyExpression(
                f"{frame_value_tracer.expr} == {format_number(self.get_value())}",
                union_free_vars(frame_value_tracer.free_vars, {"np": np}),
            ) & StringifyExpression(
                f"{frame_value_tracer.expr}.dtype == {format_dtype(self.get_value().dtype)}",
                union_free_vars(frame_value_tracer.free_vars, {"np": np}),
            )
        else:
            raise NotImplementException(
                "We can not stringify numpy variable when value is np.ndarray"
            )

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, (np.ndarray, np.number)):
            return NumpyVariable(value, graph, tracker)
        return None


class DummyVariable(VariableBase):
    """
    DummyVariable is a subclass of VariableBase used to represent a placeholder variable that has no value or reference associated with it.
    """

    def __init__(self):
        super().__init__(DanglingTracker())

    def reconstruct(self, codegen: PyCodeGen):
        codegen.gen_push_null()
