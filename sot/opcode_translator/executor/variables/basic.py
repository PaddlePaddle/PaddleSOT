from __future__ import annotations

import operator
import types
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Any

import numpy as np

import paddle

from ....infer_meta import MetaInfo
from ....symbolic.statement_ir import Symbol
from ....utils import (
    BreakGraphError,
    NameGenerator,
    NotImplementException,
    paddle_tensor_methods,
)
from ....utils.exceptions import InnerError
from ..dispatch_functions import tensor_numel
from ..guard import (
    StringifyExpression,
    check_guard,
    object_equal_stringify_guard,
    union_free_vars,
)
from ..mutable_data import MutableDictLikeData
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
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.value = value

    def get_py_value(self, allow_tensor=False):
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

    def bool_not(self):
        assert isinstance(
            self.get_py_value(), bool
        ), "Bool_not can only be applied to a bool variable."
        return VariableFactory.from_value(
            not bool(self.get_py_value()), self.graph, DummyTracker([self])
        )

    def str(self):
        return VariableFactory.from_value(
            str(self.value), self.graph, DummyTracker([self])
        )

    def format(self, *args):
        return VariableFactory.from_value(
            str(self.value).format(*[str(a.value) for a in args]),
            self.graph,
            DummyTracker([self, *args]),
        )

    def lower(self):
        return VariableFactory.from_value(
            str(self.value).lower(),
            self.graph,
            DummyTracker([self]),
        )

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if type(value) in ConstTypes:
            return ConstantVariable(value, graph, tracker)
        return None

    @staticmethod
    def wrap_literal(value: Any, graph: FunctionGraph) -> ConstantVariable:
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
        return ConstantVariable(value, graph, ConstTracker(value))


class PrintStmtVariable(VariableBase):
    def __init__(self, value: Any, graph: FunctionGraph):
        # TODO: graph should be not None
        super().__init__(None, DanglingTracker())
        self.args, self.kwargs = value
        self.graph = graph

    def _reconstruct(self, codegen: PyCodeGen):
        # do we need ? may be too strict.
        for var in self.args:
            self.graph.add_global_guarded_variable(var)
        for var in self.kwargs.values():
            self.graph.add_global_guarded_variable(var)
        # currently dont' consider kwargs
        codegen.gen_load_global("print")
        for var in self.args:
            var.reconstruct(codegen)
        codegen.gen_call_function(len(self.args))
        codegen.gen_pop_top()

    def flatten_items(self):
        return self.args


IMPLEMENTED_TENSOR_PROPERTIES = set()


def tensor_property(func):
    IMPLEMENTED_TENSOR_PROPERTIES.add(func.__name__)
    return property(func)


class DataVariable(VariableBase):
    """
    A value only object.
    If it's all magic method don't change the function_graph state, [tensor op, guard, side_effect]
    we will call it a ValueObjectVariable, we directy call python operator on it.
    """

    def __init__(
        self,
        value: Any,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.value = value

    def get_py_value(self, allow_tensor=False):
        return self.value

    make_stringify_guard = object_equal_stringify_guard

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, (paddle.dtype)):
            return DataVariable(value, graph, tracker)


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
        super().__init__(graph, tracker)
        if isinstance(tensor, paddle.Tensor):
            self.value = None
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
        self.origin_meta = self.meta
        self.var_name = TensorVariable.var_name_generator.next()

    def __len__(self):
        if self.meta.shape[0] == -1:
            raise BreakGraphError(
                "length of tensor variable with first dimension == -1"
            )
        return self.meta.shape[0]

    def get_py_value(self, allow_tensor=False):
        if allow_tensor:

            class SotTensor:
                def __init__(self, id_):
                    self.id = id_

                def __eq__(self, var):
                    if not hasattr(var, "id"):
                        return False
                    else:
                        return self.id == var.id

            return SotTensor(self.id)

        raise BreakGraphError(
            "Called TensorVariable.get_py_value. Should not use Tensor's value in simulating."
        )

    def get_py_type(self):
        return paddle.Tensor

    def get_symbol(self) -> Symbol:
        return Symbol(self.var_name)

    @property
    def out_var_name(self):
        return f"{self.graph.OUT_VAR_PREFIX}{self.var_name}"

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_fast(self.out_var_name)

    @check_guard
    def make_stringify_guard(self) -> list[StringifyExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()

        return [
            StringifyExpression(
                f"MetaInfo.from_tensor({frame_value_tracer.expr}).guard_str() == '{self.origin_meta.guard_str()}'",
                union_free_vars(
                    {"MetaInfo": MetaInfo},
                    frame_value_tracer.free_vars,
                ),
            )
        ]

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "shape": self.meta.shape,
            "dtype": DTYPE_ABBRS[self.meta.dtype],
            "stop_gradient": self.meta.stop_gradient,
            "var_name": self.var_name,
        }

    def getitem(self, key):
        return self.graph.call_tensor_method('__getitem__', self, key)

    def setitem(self, key, value):
        self.graph.add_global_guarded_variable(value)

        key_var = VariableFactory.from_value(
            key, self.graph, tracker=ConstTracker(key)
        )
        new_tensor = self.graph.call_paddle_api(
            paddle.static.setitem,
            self,
            key_var,
            value,
        )

        self.meta = new_tensor.meta
        self.graph.add_inplace_tensors(self)

    @tensor_property
    def T(self):
        """
        Return a new TensorVariable object that wraps the result of calling the transpose method on the wrapped value of this TensorVariable.
        """
        perm = list(range(len(self.meta.shape) - 1, -1, -1))
        perm_var = VariableFactory.from_value(
            perm, self.graph, tracker=ConstTracker(perm)
        )
        assert perm_var is not None
        out = self.graph.call_paddle_api(paddle.transpose, self, perm_var)
        return out

    @tensor_property
    def ndim(self):
        """
        Return a ConstantVariable object that represents the number of dimensions of the wrapped value of this TensorVariable.
        """
        self.graph.add_global_guarded_variable(self)
        return ConstantVariable.wrap_literal(len(self.meta.shape), self.graph)

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
        self.graph.add_global_guarded_variable(self)
        return ConstantVariable.wrap_literal(elements, self.graph)

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

    def numel(self):
        return self.size

    def len(self):
        if len(self.shape) == 0:
            raise InnerError("len() of a 0-D tensor is wrong")
        first_dim = self.shape[0]
        if first_dim == -1:
            raise BreakGraphError(
                "Getting len() for a dynamic shape tensor causes graph break."
            )
        self.graph.add_global_guarded_variable(self)
        return ConstantVariable.wrap_literal(first_dim, self.graph)

    def is_tensor(self):
        self.graph.add_global_guarded_variable(self)
        return ConstantVariable.wrap_literal(True, self.graph)

    def is_complex(self):
        dtype = self.meta.dtype
        is_cp_dtype = dtype in CP_DTYPE_ABBRS
        self.graph.add_global_guarded_variable(self)
        return ConstantVariable.wrap_literal(is_cp_dtype, self.graph)

    def is_integer(self):
        dtype = self.meta.dtype
        is_int_dtype = dtype in INT_DTYPE_ABBRS
        self.graph.add_global_guarded_variable(self)
        return ConstantVariable.wrap_literal(is_int_dtype, self.graph)

    def is_floating_point(self):
        dtype = self.meta.dtype
        is_fp_dtype = dtype in FP_DTYPE_ABBRS
        self.graph.add_global_guarded_variable(self)
        return ConstantVariable.wrap_literal(is_fp_dtype, self.graph)

    def getattr(self, name: str, default=None):
        if default is not None:
            raise NotImplementException(
                "default argument for getattr is not implemented"
            )
        method_name_to_builtin_fn = {
            "dim": paddle.rank,
            "numel": tensor_numel,
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
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, (paddle.Tensor, MetaInfo)):
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
        super().__init__(graph, tracker)
        self.value = obj

    make_stringify_guard = object_equal_stringify_guard

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def get_py_value(self, allow_tensor=False) -> Any:
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
        super().__init__(graph, tracker)
        self.value = slice_

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

    @cached_property
    def proxy(self):
        return self.graph.side_effects.get_proxy(
            MutableDictLikeData, self.value, self.proxy_getter
        )

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def get_py_value(self, allow_tensor=False):
        return slice(
            self.getattr("start").get_py_value(),
            self.getattr("stop").get_py_value(),
            self.getattr("step").get_py_value(),
        )

    @check_guard
    def make_stringify_guard(self) -> list[StringifyExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()
        result = (
            [
                StringifyExpression(
                    f"isinstance({frame_value_tracer.expr}, slice)",
                    frame_value_tracer.free_vars,
                ),
            ]
            + self.getattr("start").make_stringify_guard()
            + self.getattr("stop").make_stringify_guard()
            + self.getattr("step").make_stringify_guard()
        )
        return result

    def _reconstruct(self, codegen: PyCodeGen):
        # TODO(dev): Consider the case where there are tensors in the slice
        if all(
            isinstance(x, int) or x is None
            for x in [self.value.start, self.value.stop, self.value.step]
        ):
            self.graph.add_global_guarded_variable(self)
            codegen.gen_load_const(self.value)
        else:
            super()._reconstruct(codegen)

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
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
        super().__init__(graph, tracker)
        self.value = func

    def get_py_value(self, allow_tensor=False):
        return self.value

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, types.ModuleType):
            return ModuleVariable(value, graph, tracker)
        return None

    # Happened in a inline import statement.
    make_stringify_guard = object_equal_stringify_guard


class DygraphTracerVariable(VariableBase):
    # TODO(SigureMo): Remove this trick after we add CompareTracker
    def __init__(self, value, graph, tracker):
        super().__init__(graph, tracker)
        self.value = value

    def get_py_value(self, allow_tensor=False):
        return self.value

    @check_guard
    def make_stringify_guard(self) -> list[StringifyExpression]:
        return []

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "is_none": self.value is None,
        }

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
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
        super().__init__(graph, tracker)
        self.value = value

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def get_py_value(self, allow_tensor=False) -> Any:
        return self.value

    @check_guard
    def make_stringify_guard(self) -> list[StringifyExpression]:
        if isinstance(self.get_py_value(), np.number):
            frame_value_tracer = self.tracker.trace_value_from_frame()

            def format_dtype(dtype: np.dtype):
                return f"np.{str(dtype)}"

            def format_number(number: np.number):
                return f"{format_dtype(number.dtype)}({str(number.item())})"

            return [
                StringifyExpression(
                    f"{frame_value_tracer.expr} == {format_number(self.get_py_value())}",
                    union_free_vars(frame_value_tracer.free_vars, {"np": np}),
                ),
                StringifyExpression(
                    f"{frame_value_tracer.expr}.dtype == {format_dtype(self.get_py_value().dtype)}",
                    union_free_vars(frame_value_tracer.free_vars, {"np": np}),
                ),
            ]
        else:
            raise NotImplementException(
                "We can not stringify numpy variable when value is np.ndarray"
            )

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, (np.ndarray, np.number)):
            return NumpyVariable(value, graph, tracker)
        return None


class DummyVariable(VariableBase):
    """
    DummyVariable is a subclass of VariableBase used to represent a placeholder variable that has no value or reference associated with it.
    """

    def __init__(self):
        # TODO: graph should be not None
        super().__init__(None, DanglingTracker())

    def reconstruct(self, codegen: PyCodeGen):
        codegen.gen_push_null()


class CellVariable(VariableBase):
    def __init__(self, value=None):
        # TODO: graph should be not None
        super().__init__(
            None, DanglingTracker()
        )  # should reconstruct cell variable
        assert isinstance(value, (VariableBase, type(None)))
        self.set_value(value)

    def cell_content(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def empty(self):
        return self.value is None
