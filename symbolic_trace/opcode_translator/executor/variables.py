from __future__ import annotations

import inspect
import types
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable

import paddle

from ...infer_meta import MetaInfo
from ...proxy_tensor import ProxyTensor, ProxyTensorContext
from ...symbolic.statement_ir import Symbol
from ...utils import ASSERT, NameGenerator, is_paddle_api, log_do
from ...utils.exceptions import InnerError
from .pycode_generator import PyCodeGen
from .tracker import (
    ConstTracker,
    DummyTracker,
    GetAttrTracker,
    GetItemTracker,
    Tracker,
)

if TYPE_CHECKING:
    from .function_graph import FunctionGraph

Guard = Callable[[types.FrameType], bool]
ConstTypes = (int, float, str, bool, type(None))


def compose_guards(guards: list[Guard]) -> Guard:
    def composed_guard_fn(frame: types.FrameType) -> bool:
        ret = True
        for guard in guards:
            ret = ret and guard(frame)
        return ret

    return composed_guard_fn


def get_zero_degree_vars(
    variables: set[VariableTracker], visited_vars: list[VariableTracker]
) -> list[VariableTracker]:
    return [
        var
        for var in variables
        if var not in visited_vars
        and len(set(var.get_traceable_inputs()) - set(visited_vars)) == 0
    ]


def topo_sort_vars(
    root_variables: list[VariableTracker],
) -> list[VariableTracker]:
    variables = set()

    for root in root_variables:
        variables.add(root)
        variables |= set(root.flatten_traceable_inputs())

    topo_ordered_vars = []
    topo_queue = Queue()
    for var in get_zero_degree_vars(variables, topo_ordered_vars):
        topo_queue.put(var)

    while not topo_queue.empty():
        var = topo_queue.get()
        topo_ordered_vars.append(var)
        for zero_degree_var in get_zero_degree_vars(
            variables, topo_ordered_vars
        ):
            if (
                zero_degree_var in topo_queue.queue
                or zero_degree_var in topo_ordered_vars
            ):
                continue
            topo_queue.put(zero_degree_var)
    return topo_ordered_vars


class VariableTrackerFactory:
    registered_funcs: list[Callable] = []

    @staticmethod
    def default_from_value(value, graph, tracker):
        return ObjectVariable(value, graph, tracker)
        raise RuntimeError(
            f"Don't Implement a value binding method for type: `{type(value)}`"
        )

    @staticmethod
    def register_from_value(from_value_func: Callable):
        VariableTrackerFactory.registered_funcs.append(from_value_func)

    @staticmethod
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        for func in VariableTrackerFactory.registered_funcs:
            var = func(value, graph, tracker)
            if var is not None:
                return var
        return VariableTrackerFactory.default_from_value(value, graph, tracker)


class VariableTracker:
    """
    we first deal guard information collection.
    """

    tracker: Tracker
    name_generator = NameGenerator("tracker_")

    def __init__(self, tracker: Tracker):
        self.tracker = tracker
        self.id = VariableTracker.name_generator.next()

    def __hash__(self):
        return hash(self.id)

    def make_check_fn(self) -> Guard:
        assert not isinstance(
            self.tracker, DummyTracker
        ), "Can not make guard from dummy tracker"

        def guard_fn(frame: types.FrameType) -> bool:
            value = self.tracker.trace_value_from_frame()(frame)
            log_do(
                3,
                lambda: print(
                    f"[Guard]: guard_fn for {self}, tracker={self.tracker.__class__.__name__}, value={value}"
                ),
            )
            if isinstance(self, TensorVariable):
                return MetaInfo.from_tensor(value) == self.get_value().meta
            return self.get_value() == value

        return guard_fn

    def get_value(self) -> Any:
        raise NotImplementedError()

    def reconstruct(self, codegen: PyCodeGen):
        if (
            not isinstance(self.tracker, DummyTracker)
            and self.tracker.is_traceable()
        ):
            self.tracker.gen_instructions(codegen)
        else:
            self._reconstruct(codegen)

    def _reconstruct(self, codegen: PyCodeGen):
        raise NotImplementedError()

    def flatten_items(self) -> list[VariableTracker]:
        if not isinstance(self, ContainerVariable):
            return [self]
        flattened_items = []
        for item in self.get_items():
            flattened_items.extend(item.flatten_items())
        return flattened_items

    def get_inputs(self) -> list[VariableTracker]:
        return self.tracker.inputs

    def get_traceable_inputs(self) -> list[VariableTracker]:
        if self.tracker.is_traceable:
            return []

        return list(
            filter(lambda x: x.tracker.is_traceable(), self.tracker.inputs)
        )

    def flatten_traceable_inputs(self) -> list[VariableTracker]:
        flattened_traceable_inputs: list[VariableTracker] = [self]
        if self.tracker.is_traceable:
            return flattened_traceable_inputs

        for input in self.get_inputs():
            flattened_traceable_inputs.extend(input.flatten_traceable_inputs())
        return flattened_traceable_inputs

    def call_function(self, *args, **kwargs):
        pass

    def getattr(self, *args, **kwargs):
        pass

    def getitem(self, *args, **kwargs):
        pass

    @VariableTrackerFactory.register_from_value
    def from_value(
        value: Any,
        graph: FunctionGraph | None,
        tracker: Tracker,
    ):
        if isinstance(value, VariableTracker):
            return value
        return None


class ConstantVariable(VariableTracker):
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
        var = VariableTrackerFactory.from_value(
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
        var = VariableTrackerFactory.from_value(
            operator(other.value), None, tracker=DummyTracker([self, other])
        )
        return var

    @VariableTrackerFactory.register_from_value
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


class TensorVariable(VariableTracker):
    def __init__(
        self,
        tensor: paddle.Tensor | ProxyTensor,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.leaf = False
        if isinstance(tensor, paddle.Tensor):
            self.value = ProxyTensorContext().from_tensor(tensor)
            self.leaf = True
        elif isinstance(tensor, ProxyTensor):
            self.value = tensor
        self.graph = graph

    def get_value(self):
        return self.value

    def get_symbol(self) -> Symbol:
        return Symbol(self.value.name)

    @property
    def out_var_name(self):
        return f"{self.graph.out_var_prefix}{self.value.name}"

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_fast(self.out_var_name)

    # Paddle variable do not have inplace operators. For example when call `y **= x`, will call var.__pow__.
    # If inplace operator do not impl, it will try to call non-inplace operator, so we do not impl inplace magic method here

    def __repr__(self) -> str:
        return f"TensorVariable{self.value.meta}"

    def __getattr__(self, name: str):
        # TODO: Handle attribute case
        return TensorMethodVariable(
            self, name, self.graph, tracker=GetAttrTracker(self, name)
        )

    @VariableTrackerFactory.register_from_value
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, (paddle.Tensor, ProxyTensor)):
            assert graph is not None
            return TensorVariable(value, graph, tracker)
        return None


class ContainerVariable(VariableTracker):
    def get_items(self) -> list[VariableTracker]:
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __bool__(self):
        return len(self) > 0


class ListVariable(ContainerVariable):
    def __init__(
        self,
        val_list: list[VariableTracker],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.graph = graph
        # everything in stack is VariableTracker, so just accept the input list is ok
        self.value = val_list

    def get_value(self):
        return [self[i].get_value() for i in range(len(self))]

    def _reconstruct(self, codegen: PyCodeGen):
        size = len(self)
        for idx in range(size):
            self[idx].reconstruct(codegen)
        codegen.gen_build_list(size)

    def get_items(self):
        size = len(self)
        return [self[idx] for idx in range(size)]

    def get_wrapped_items(self):
        return self.get_items()

    def __repr__(self) -> str:
        return f"ListVariable(len={len(self)})"

    def __len__(self):
        return len(self.value)

    def __getitem__(self, key):
        '''
        we need to make sure that:
            before an inplace change happens to ListVariable,
            the related items should already be wrapped as VariableTracker

        if not, tracker might be set to a wrong elem
        '''
        if isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        retval = self.value[key]

        # if list is an input of funciton, we need make sure __getitem__ returns a VariableTracker
        retval = VariableTrackerFactory.from_value(
            retval, self.graph, tracker=GetItemTracker(self, key)
        )

        return retval

    def __setitem__(self, key, value):
        '''
        why __setitem__ is ok:

        case:
            def f(x = [t0, t1])
                ...
                x[0] = 0
                ...

            1. if setitem happens after get t0: t0 is a VariableTracker (transformed at getitem), so it is ok
            2. if setitem happens before get t0: t0 will not be used
        '''
        if isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} as key."
            )

        if not isinstance(value, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {value} to set value."
            )
        self.value[key] = value

    def __delitem__(self, key):
        if isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} as key to delete."
            )
        del self.value[key]

    @VariableTrackerFactory.register_from_value
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, list):
            assert graph is not None
            return ListVariable(value, graph=graph, tracker=tracker)
        return None


class TupleVariable(ContainerVariable):
    def __init__(
        self,
        val_tuple: list[VariableTracker],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.graph = graph
        # exactly it is a list (need replace item with VariableTracker)
        self.value = list(val_tuple)

    def get_value(self):
        return tuple(self[i].get_value() for i in range(len(self)))

    def _reconstruct(self, codegen: PyCodeGen):
        size = len(self)
        for idx in range(size):
            self[idx].reconstruct(codegen)
        codegen.gen_build_tuple(size)

    def get_items(self):
        size = len(self)
        return [self[idx] for idx in range(size)]

    def get_wrapped_items(self):
        return self.get_items()

    def __repr__(self) -> str:
        return f"TupleVariable(len={len(self)})"

    def __len__(self):
        return len(self.value)

    def __getitem__(self, key):
        if isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )
        retval = self.value[key]

        return VariableTrackerFactory.from_value(
            retval, graph=self.graph, tracker=GetItemTracker(self, key)
        )

    def __setitem__(self, key, value):
        raise InnerError(
            f"[{self.__class__.__name__}]: setitem is not allowed."
        )

    def __delitem__(self, key):
        raise InnerError(
            f"[{self.__class__.__name__}]: delitem is not allowed."
        )

    @VariableTrackerFactory.register_from_value
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, tuple):
            return TupleVariable(value, graph, tracker)
        return None


class DictVariable(ContainerVariable):
    def __init__(
        self,
        val_dict: dict[object, VariableTracker],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.graph = graph
        self.value = val_dict

    def get_value(self):
        return {key: self[key].get_value() for key in self.value}

    def _reconstruct(self, codegen: PyCodeGen):
        size = len(self)
        for key in self.value.keys():
            if not isinstance(key, ConstTypes):
                raise InnerError(
                    f"[{self.__class__.__name__}]: recieved {key} as key."
                )
            key_var = ConstantVariable.wrap_literal(key)
            value_var = self[key]
            key_var.reconstruct(codegen)
            value_var.reconstruct(codegen)
        codegen.gen_build_map(size)

    def get_items(self):
        items = []
        for key in self.value.keys():
            if not isinstance(key, ConstTypes):
                raise InnerError(
                    f"[{self.__class__.__name__}]: recieved {key} as key."
                )
            key_var = VariableTrackerFactory.from_value(
                key, self.graph, tracker=ConstTracker(key)
            )
            value_var = self[key]
            items.extend([key_var, value_var])
        return items

    def get_wrapped_items(self):
        items = {}
        for key in self.value.keys():
            if not isinstance(key, ConstTypes):
                raise InnerError(
                    f"[{self.__class__.__name__}]: recieved {key} as key."
                )
            items[key] = self[key]
        return items

    def __repr__(self) -> str:
        return f"DictVariable(len={len(self)})"

    def __len__(self):
        return len(self.value)

    def __getitem__(self, key):
        if isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        retval = self.value[key]

        return VariableTrackerFactory.from_value(
            retval, self.graph, tracker=GetItemTracker(self, key)
        )

    def __setitem__(self, key, value):
        if isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key."
            )

        if not isinstance(value, ConstantVariable):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {value} to set value."
            )

        self.value[key] = value

    def __delitem__(self, key):
        if isinstance(key, VariableTracker):
            raise InnerError(
                f"[{self.__class__.__name__}]: recieved {key} as key to delete."
            )
        del self.value[key]

    @VariableTrackerFactory.register_from_value
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, dict):
            assert graph is not None
            return DictVariable(value, graph=graph, tracker=tracker)


class CallableVariable(VariableTracker):
    def __init__(self, graph: FunctionGraph, tracker: Tracker):
        super().__init__(tracker)
        self.graph = graph

    def __call__(self, *args, **kwargs) -> VariableTracker:
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


class PaddleApiVariable(FunctionVariable):
    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(fn, graph, tracker)

    def call_function(self, *args, **kwargs):
        return self.graph.call_paddle_api(self.value, *args, **kwargs)

    @VariableTrackerFactory.register_from_value
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        # This should be front of FunctionVariable to avoid conflict.
        if callable(value) and is_paddle_api(value):
            return PaddleApiVariable(value, graph, tracker)
        return None

    def __repr__(self) -> str:
        return f"PaddleApiVariable({self.value.__name__})"


class UserDefinedFunctionVariable(FunctionVariable):
    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(fn, graph, tracker)

    def call_function(self, *args, **kwargs) -> VariableTracker:
        from .opcode_inline_executor import OpcodeInlineExecutor

        if self.value is ASSERT:
            return self.value(args[0].value)

        inline_executor = OpcodeInlineExecutor(self, *args, **kwargs)
        output = inline_executor.inline_call()
        return output

    @VariableTrackerFactory.register_from_value
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, (types.FunctionType)):
            return UserDefinedFunctionVariable(value, graph, tracker)
        return None

    def __repr__(self) -> str:
        return f"UserDefinedFunctionVariable({self.value.__name__})"


class MethodVariable(CallableVariable):
    def __init__(
        self,
        bound_instance: VariableTracker,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.bound_instance = bound_instance


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
        return self.graph.call_tensor_method(
            self.method_name, self.tensor, *args, **kwargs
        )

    @VariableTrackerFactory.register_from_value
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

    @VariableTrackerFactory.register_from_value
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if inspect.ismethod(value):
            method_self = VariableTrackerFactory.from_value(
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


class LayerVariable(CallableVariable):
    def __init__(
        self, layer: paddle.nn.Layer, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(graph, tracker)
        self.value = layer

    def get_value(self):
        return self.value

    def __getattr__(self, name: str):
        if not hasattr(self.value, name):
            raise InnerError(f"LayerVariable {self} has no attribute {name}")
        attr = getattr(self.value, name)
        if inspect.ismethod(attr):
            return UserDefinedMethodVariable(
                self, attr.__func__, self.graph, self.tracker
            )
        return VariableTrackerFactory.from_value(
            attr, self.graph, tracker=GetAttrTracker(self, name)
        )


class PaddleLayerVariable(LayerVariable):
    def __init__(
        self, layer: paddle.nn.Layer, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(layer, graph, tracker)
        self.name = self.graph.sir_ctx.new_layername()

    def get_symbol(self) -> Symbol:
        return Symbol(self.name)

    def call_function(self, *args, **kwargs):
        return self.graph.call_layer(self, *args, **kwargs)

    @VariableTrackerFactory.register_from_value
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, paddle.nn.Layer) and value.__module__.startswith(
            "paddle.nn."
        ):
            return PaddleLayerVariable(value, graph, tracker)
        return None

    def __repr__(self) -> str:
        return f"PaddleLayerVariable({self.value.__class__.__name__})"


class UserDefinedLayerVariable(LayerVariable):
    def __init__(
        self, layer: paddle.nn.Layer, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(layer, graph, tracker)

    def call_function(self, *args, **kwargs):
        fn_var = UserDefinedFunctionVariable(
            self.value.__class__.__call__, self.graph, self.tracker
        )

        return fn_var(*(self, *args), **kwargs)

    @VariableTrackerFactory.register_from_value
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
        return self.value(*args, **kwargs)

    @VariableTrackerFactory.register_from_value
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, (types.BuiltinFunctionType)):
            return BuiltinVariable(value, graph, tracker)
        return None

    def __repr__(self) -> str:
        return f"BuiltinVariable({self.value.__name__})"


class SliceVariable(VariableTracker):
    def __init__(self, slice_, graph, tracker):
        super().__init__(tracker)
        self.value = slice_
        self.graph = graph

    def __repr__(self) -> str:
        return f"SliceVariable({self.value})"

    @VariableTrackerFactory.register_from_value
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, slice):
            return SliceVariable(value, graph, tracker)
        return None


class ModuleVariable(VariableTracker):
    def __init__(self, func, graph, tracker):
        super().__init__(tracker)
        self.value = func
        self.graph = graph

    def get_value(self):
        return self.value

    def __getattr__(self, name: str):
        if not hasattr(self.value, name):
            raise InnerError(f"ModuleVariable {self} has no attribute {name}")
        attr = getattr(self.value, name)
        return VariableTrackerFactory.from_value(
            attr, self.graph, tracker=GetAttrTracker(self, name)
        )

    @VariableTrackerFactory.register_from_value
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, types.ModuleType):
            return ModuleVariable(value, graph, tracker)
        return None


class ObjectVariable(VariableTracker):
    def __init__(self, obj, graph, tracker):
        super().__init__(tracker)
        self.value = obj
        self.graph = graph

    def __repr__(self) -> str:
        return f"ObjectVariable({self.value})"

    def __getattr__(self, name: str):
        if not hasattr(self.value, name):
            raise InnerError(f"ObjectVariable {self} has no attribute {name}")
        attr = getattr(self.value, name)
        return VariableTrackerFactory.from_value(
            attr, self.graph, tracker=GetAttrTracker(self, name)
        )
