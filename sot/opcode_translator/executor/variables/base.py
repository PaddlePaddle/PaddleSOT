from __future__ import annotations

import inspect
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable, Optional

import paddle

from ....utils import NameGenerator, get_unbound_method, log, log_do
from ....utils.exceptions import InnerError, NotImplementException
from ..guard import StringifyExpression, union_free_vars
from ..pycode_generator import PyCodeGen
from ..tracker import DummyTracker, GetAttrTracker, GetItemTracker, Tracker

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph

    FromValueFunc = Callable[
        [Any, Optional[FunctionGraph], Tracker], Optional["VariableBase"]
    ]


ConstTypes = (int, float, str, bool, type(None))


def get_zero_degree_vars(
    variables: set[VariableBase], visited_vars: list[VariableBase]
) -> list[VariableBase]:
    return [
        var
        for var in variables
        if var not in visited_vars
        and len(set(var.get_traceable_inputs()) - set(visited_vars)) == 0
    ]


def topo_sort_vars(
    root_vars: list[VariableBase],
) -> list[VariableBase]:
    unique_vars = set()

    for var in root_vars:
        unique_vars |= set(var.flatten_traceable_inputs())

    topo_ordered_vars = []
    topo_queue = Queue()
    for var in get_zero_degree_vars(unique_vars, topo_ordered_vars):
        topo_queue.put(var)

    while not topo_queue.empty():
        var = topo_queue.get()
        topo_ordered_vars.append(var)
        for zero_degree_var in get_zero_degree_vars(
            unique_vars, topo_ordered_vars
        ):
            if (
                zero_degree_var in topo_queue.queue
                or zero_degree_var in topo_ordered_vars
            ):
                continue
            topo_queue.put(zero_degree_var)
    return topo_ordered_vars


def map_variables(map_func, variables):
    def _map_variable(variable):
        assert isinstance(
            variable, VariableBase
        ), f"variable must be VariableBase, got {variable}"
        from .container import ContainerVariable

        if isinstance(variable, ContainerVariable):
            return paddle.utils.map_structure(
                _map_variable, variable.get_wrapped_items()
            )
        return map_func(variable)

    return paddle.utils.map_structure(_map_variable, variables)


class VariableFactory:
    """
    A factory class for creating variables from arbitrary values.

    This class provides a set of registration and factory methods for creating variables
    of different types based on the type of the input value.

    """

    registered_funcs: dict[str, list[str]] = {"default": []}
    mapping_str_func: dict[str, FromValueFunc] = {}

    @staticmethod
    def default_from_value(value, graph, tracker):
        """
        A default factory function that creates an ObjectVariable from the given value.

        Args:
            value: The input value.
            graph: The FunctionGraph object that this variable is associated with.
            tracker: The Tracker object that tracks the information of this variable.

        Returns:
            ObjectVariable: A new ObjectVariable representing the input value.
        """
        from .basic import ObjectVariable

        return ObjectVariable(value, graph, tracker)

    @staticmethod
    def register_from_value(*, successor: str | None = None):
        """
        A decorator that registers a from_value function with the VariableFactory class.

        Args:
            successor (str | None): The name of the successor registration group, or None if this is the default group.

        Returns:
            Callable: A wrapper function that registers the decorated function with the VariableFactory class.
        """
        registered_funcs = VariableFactory.registered_funcs
        mapping_str_func = VariableFactory.mapping_str_func

        def _register_from_value(func: FromValueFunc):
            name = func.__qualname__.split(".")[0]
            mapping_str_func[name] = func
            if successor is None:
                registered_funcs["default"].append(name)
            elif successor not in registered_funcs.keys():
                registered_funcs[successor] = [name]
            else:
                registered_funcs[successor].append(name)

        log(4, VariableFactory.registered_funcs)
        return _register_from_value

    @staticmethod
    def from_value(
        value: Any,
        graph: FunctionGraph | None,
        tracker: Tracker,
        *,
        debug_name: str | None = None,
    ):
        """
        Create a new variable object from the given value.

        This method searches through the registered from_value functions to find one
        that can create a variable object from the given value. If no matching function
        is found, the default_from_value function is used.

        Args:
            value (Any): The input value.
            graph (FunctionGraph | None): The FunctionGraph object that this variable is associated with.
            tracker (Tracker): The Tracker object that tracks the information of this variable.
            debug_name (str | None): An optional debug name for the variable.

        Returns:
            VariableBase: A new variable object representing the input value.
        """
        registered_funcs = VariableFactory.registered_funcs

        def _find_var(key: str = "default"):
            for name in registered_funcs[key]:
                if name in registered_funcs.keys():
                    var = _find_var(name)
                    if var is not None:
                        return var
                func = VariableFactory.mapping_str_func[name]
                var = func(value, graph, tracker)
                if var is not None:
                    return var

        var = _find_var()
        if var is None:
            var = VariableFactory.default_from_value(value, graph, tracker)
        var.debug_name = debug_name
        return var


class VariableBase:
    """
    VariableBase is a basic concept and each symbols in VM stack is regarded as
    an Variable Object in symblic tracing process.

    It is designed to wrap the variables traced by tracker.

    Args:
        tracker(Tracker): The Tracker object that tracks the information of this variable.

    Note:
        It serves as an abstract class and should not be instantiated directly.
    """

    tracker: Tracker
    name_generator = NameGenerator("object_")

    def __init__(self, tracker: Tracker):
        self.tracker = tracker
        self.id = VariableBase.name_generator.next()
        self._debug_name: str | None = None

    @property
    def main_info(self) -> dict[str, Any]:
        """
        Wrap the variable with relevant information

        Returns:
            main_info: Main information of the variable.
        """
        return {}

    @property
    def debug_info(self) -> dict[str, Any]:
        """
        Wrap the debug-related information of the variable.

        Returns:
            debug_name: The name of variable.
            id: The id of variable.
        """
        return {
            "debug_name": self.debug_name,
            "id": self.id,
        }

    @property
    def debug_name(self) -> str:
        """
        Generate a debug_name for each variable.

        Returns:
            _debug_name: the name of variable.
        """
        if self._debug_name is None:
            inputs = self.tracker.inputs
            if isinstance(self.tracker, GetItemTracker):
                return (
                    f"{self.tracker.container.debug_name}[{self.tracker.key}]"
                )
            elif isinstance(self.tracker, GetAttrTracker):
                return f"{self.tracker.obj.debug_name}.{self.tracker.attr}"
            else:
                # TODO: refine the debug_name for other trackers
                if len(inputs) == 0:
                    self._debug_name = "tmp_var"
                else:
                    for input in inputs:
                        assert input is not None
                    self._debug_name = "tmp_var_" + "_".join(
                        input.debug_name for input in inputs
                    )
        return self._debug_name

    @debug_name.setter
    def debug_name(self, name):
        self._debug_name = name

    def __hash__(self):
        return hash(self.id)

    def make_stringify_guard(self) -> StringifyExpression:
        """
        Create a StringifyExpression object that represents a guard expression for this variable.

        Returns:
            StringifyExpression: An object that contains the guard expression and the free variables used in the expression.
        """
        assert (
            self.tracker.is_traceable()
        ), "Cannot make guard from a non-traceable variable."

        frame_value_tracer = self.tracker.trace_value_from_frame()
        log_do(
            4,
            lambda: print(
                f"[Guard]: guard_fn for {self}, tracker={self.tracker.__class__.__name__}, value={frame_value_tracer.expr}"
            ),
        )
        return StringifyExpression(
            f"{frame_value_tracer.expr} == {self.get_value()!r}",
            union_free_vars(frame_value_tracer.free_vars),
        )

    def get_value(self) -> Any:
        raise NotImplementedError()

    def get_type(self):
        return type(self.get_value())

    def reconstruct(self, codegen: PyCodeGen):
        """
        Contruct an opcode and append it into codegen.instructions.
        """
        if (
            not isinstance(self.tracker, DummyTracker)
            and self.tracker.is_traceable()
        ):
            self.tracker.gen_instructions(codegen)
        else:
            self._reconstruct(codegen)

    def _reconstruct(self, codegen: PyCodeGen):
        raise NotImplementException()

    def flatten_items(self) -> list[VariableBase]:
        """
        Recursively flatten the items in this container variable to a list of Variable objects.

        Returns:
            list[VariableBase]: A list of Variable objects.
        """
        from .container import ContainerVariable

        if not isinstance(self, ContainerVariable):
            return [self]
        flattened_items = []
        for item in self.get_items():
            flattened_items.extend(item.flatten_items())
        return flattened_items

    def get_inputs(self) -> list[VariableBase]:
        return self.tracker.inputs

    def get_traceable_inputs(self) -> list[VariableBase]:
        if self.tracker.is_traceable():
            return []

        return list(
            filter(lambda x: x.tracker.is_traceable(), self.tracker.inputs)
        )

    def flatten_traceable_inputs(self) -> list[VariableBase]:
        """
        Recursively flatten the items in this traceable_inputs to a list of Variable objects.

        Returns:
            list[Variable]: A list of Variable objects.
        """
        if self.tracker.is_traceable():
            return [self]

        flattened_traceable_inputs: list[VariableBase] = []
        for input in self.get_inputs():
            flattened_traceable_inputs.extend(input.flatten_traceable_inputs())
        return flattened_traceable_inputs

    def call_function(self, *args, **kwargs):
        pass

    def getattr(self, name: str, default=None):
        """
        Get the value of an attribute with the given name from the underlying object of this variable.

        Args:
            name(str): The name of the attribute to retrieve.

        Returns:
            Variable object: A new variable representing the value of the requested attribute,
                             or a MethodVariable object if the attribute is a method.
        """
        if not hasattr(self.value, name):
            if default is not None:
                assert isinstance(default, VariableBase)
                return default
            raise InnerError(
                f"{self.__class__.__name__} {self} has no attribute {name}"
            )
        attr = getattr(self.value, name)
        if inspect.ismethod(attr):
            from .callable import MethodVariable

            return MethodVariable.wrap_method(
                value=attr,
                instance=self,
                graph=self.graph,
                tracker=GetAttrTracker(self, name),
                method_name=name,
            )

        return VariableFactory.from_value(
            attr, self.graph, tracker=GetAttrTracker(self, name)
        )

    def __setitem__(self, key, value):
        return self.setitem(key, value)

    def setitem(self, key, value):
        raise NotImplementException(f"{self} is not support setitem.")

    def __repr__(self):
        info = {**self.main_info, **self.debug_info}
        info_str = ", ".join([f"{value}" for value in info.values()])
        return f"{self.__class__.__name__}({info_str})"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, idx):
        return self.getitem(idx)

    def getitem(self, item):
        class_var = VariableFactory.from_value(
            self.get_value().__class__,
            self.graph,
            GetAttrTracker(self, '__class__'),
        )
        fn_var = VariableFactory.from_value(
            get_unbound_method(self.get_value(), '__getitem__'),
            self.graph,
            GetAttrTracker(class_var, '__getitem__'),
        )
        output = fn_var(item)
        return output

    def __call__(self, *args, **kwargs):
        """
        Call the object represented by this variable with the given arguments.

        Args:
            *args: Positional arguments to pass to the object's __call__ method.
            **kwargs: Keyword arguments to pass to the object's __call__ method.

        Returns:
            VariableBase: A new variable representing the result of calling the object's __call__ method.
        """
        from .callable import BuiltinVariable, UserDefinedFunctionVariable

        class_var = VariableFactory.from_value(
            self.get_value().__class__,
            self.graph,
            GetAttrTracker(self, '__class__'),
        )
        unbound_method = get_unbound_method(self.get_value(), '__call__')
        if hasattr(unbound_method, "__code__"):
            fn_var = UserDefinedFunctionVariable(
                unbound_method,
                self.graph,
                GetAttrTracker(class_var, '__call__'),
            )
        else:
            fn_var = BuiltinVariable(
                unbound_method,
                self.graph,
                GetAttrTracker(class_var, '__call__'),
            )
        output = fn_var(*args, **kwargs)
        return output

    @VariableFactory.register_from_value()
    def from_value(
        value: Any,
        graph: FunctionGraph | None,
        tracker: Tracker,
    ):
        if isinstance(value, VariableBase):
            return value
        return None
