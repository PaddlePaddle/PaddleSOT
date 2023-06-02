from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import paddle

from ....infer_meta import MetaInfo
from ....symbolic.statement_ir import Symbol
from ....utils import NameGenerator, log_do, paddle_tensor_methods
from ....utils.exceptions import InnerError
from ..guard import StringifyExpression, union_free_vars
from ..pycode_generator import PyCodeGen
from ..tracker import ConstTracker, DummyTracker, GetAttrTracker, Tracker
from .base import ConstTypes, VariableBase, VariableFactory

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph


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

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_const(self.value)

    def __repr__(self) -> str:
        return f"ConstantVariable({self.value})"

    def __bool__(self) -> bool:
        return bool(self.value)

    def apply_unary_operator(self, magic_name):
        operator = getattr(self.value, magic_name)
        var = VariableFactory.from_value(
            operator(),
            None,
            tracker=DummyTracker(
                [
                    self,
                ]
            ),
        )
        return var

    def apply_binary_operator(self, other, magic_name):
        if not isinstance(other, ConstantVariable):
            return NotImplemented
        operator = getattr(self.value, magic_name)
        var = VariableFactory.from_value(
            operator(other.value), None, tracker=DummyTracker([self, other])
        )
        return var

    @VariableFactory.register_from_value(before="VariableBase")
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

    def __repr__(self) -> str:
        return f"TensorVariable{self.meta}"

    def __getitem__(self, key):
        return self.graph.call_tensor_method(
            '__getitem__',
            self,
            VariableFactory.from_value(
                key, self.graph, tracker=ConstTracker(key)
            ),
        )

    @property
    def T(self):
        perm = list(range(len(self.meta.shape) - 1, -1, -1))
        perm_var = VariableFactory.from_value(
            perm, self.graph, tracker=ConstTracker(perm)
        )
        out = self.graph.call_paddle_api(paddle.transpose, self, perm_var)
        return out

    @property
    def ndim(self):
        return ConstantVariable.wrap_literal(len(self.meta.shape))

    def __getattr__(self, name: str):
        if name in paddle_tensor_methods:
            from .callable import TensorMethodVariable

            return TensorMethodVariable(
                self, name, self.graph, tracker=GetAttrTracker(self, name)
            )
        elif name in ["shape", "dtype", "stop_gradient"]:
            return VariableFactory.from_value(
                getattr(self.meta, name),
                self.graph,
                tracker=GetAttrTracker(self, name),
            )
        elif name in ["T", "ndim"]:
            return getattr(self, name)
        else:
            raise InnerError(f"Unknown Tensor attribute: {name}")

    @VariableFactory.register_from_value(before="VariableBase")
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

    def __repr__(self) -> str:
        return f"ObjectVariable({self.value})"


class SliceVariable(VariableBase):
    def __init__(self, slice_, graph, tracker):
        super().__init__(tracker)
        self.value = slice_
        self.graph = graph

    def __repr__(self) -> str:
        return f"SliceVariable({self.value})"

    def get_value(self):
        return self.value

    @VariableFactory.register_from_value(before="VariableBase")
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

    def __repr__(self) -> str:
        return f"ModuleVariable({self.value})"

    @VariableFactory.register_from_value(before="VariableBase")
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

    def __repr__(self) -> str:
        return f"DygraphTracerVariable(is_none={self.value is None})"

    @VariableFactory.register_from_value(before="VariableBase")
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, paddle.fluid.dygraph.tracer.Tracer):
            return DygraphTracerVariable(value, graph, tracker)
        return None
