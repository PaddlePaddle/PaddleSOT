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

    # Each variable object should implement a method called `from_value`,
    # which should adhere to the FromValueFunc signature.
    FromValueFunc = Callable[
        [Any, Optional[FunctionGraph], Tracker], Optional["VariableBase"]
    ]


ConstTypes = (int, float, str, bool, type(None))


def analyse_traceable_vars(
    root: VariableBase,
    degree: dict[VariableBase, int],
    to_parents: dict[VariableBase, set[VariableBase]],
    topo_queue: Queue[VariableBase],
) -> None:
    if root in degree:
        return

    inputs = root.get_inputs()
    traceable_inputs = (
        root.get_traceable_inputs() if not root.tracker.is_traceable() else []
    )

    if not root.tracker.is_traceable():
        for var in inputs:
            analyse_traceable_vars(var, degree, to_parents, topo_queue)
    else:
        degree[root] = len(traceable_inputs)
        if len(traceable_inputs) == 0:
            topo_queue.put(root)

        for var in inputs:
            if var not in to_parents:
                to_parents[var] = set()
            to_parents[var].add(root)


def topo_sort_vars(
    root_vars: list[VariableBase],
) -> list[VariableBase]:
    """
    This function is used to sort the input variables in a topological order.
    Args:
        root_vars (list[VariableBase]): A list of root variables from which the ordering starts.

    Returns:
        list[VariableBase]: A list of variables in topological order.
    """
    retval: list[VariableBase] = []

    degree: dict[VariableBase, int] = {}
    to_parents: dict[VariableBase, set[VariableBase]] = {}
    topo_queue: Queue[VariableBase] = Queue()

    for var in root_vars:
        analyse_traceable_vars(var, degree, to_parents, topo_queue)

    while not topo_queue.empty():
        var = topo_queue.get()
        retval.append(var)
        if var in to_parents:
            for parent in to_parents[var]:
                degree[parent] -= 1
                if degree[parent] == 0:
                    topo_queue.put(parent)

    return retval


def map_variables(map_func, variables: list[VariableBase]):
    """
    This function maps the given map_func to the given list of variables in a recursive manner.
    Args:
        map_func (Callable[[VariableBase], Any]): The function to be mapped to each variable.
        variables (list[VariableBase]): A list of variables to which the map_func is to be applied.

    Returns:
        tuple: The result of applying the map_func to the variables.
    """

    def _map_variable(variable: VariableBase):
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
        A decorator function that registers a function for creating a Variable from a value.

        Args:
            successor (str | None, optional): The name of the successor function that will be called after this function when creating a Variable. If None, the function is added to a default list of functions.

        Returns:
            The _register_from_value decorator function, which takes the function to be registered as an argument.
        """
        registered_funcs = VariableFactory.registered_funcs
        mapping_str_func = VariableFactory.mapping_str_func

        def _register_from_value(func: FromValueFunc):
            """
            Function to register a function for creating a Variable from a value
            """
            # Get the name of the function
            name = func.__qualname__.split(".")[0]
            # Map the name of the function to the function
            mapping_str_func[name] = func
            if successor is None:
                registered_funcs["default"].append(
                    name
                )  # If successor is None, add the function to the "default" list
            elif successor not in registered_funcs.keys():
                registered_funcs[successor] = [
                    name
                ]  # If the successor is not in the registered_funcs dictionary, set the value to a list containing only name
            else:
                registered_funcs[successor].append(
                    name
                )  # If the successor is in the registered_funcs dictionary, append name to the existing list of functions for that successor

        log(
            4, VariableFactory.registered_funcs
        )  # Print the registered_funcs dictionary if the logging level is at least 4
        return _register_from_value

    @staticmethod
    def from_value(
        value: Any,
        graph: FunctionGraph | None,
        tracker: Tracker,
        *,
        debug_name: str | None = None,
    ) -> VariableBase | None:
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
                    # If the function name is a key in the registered_funcs dictionary, recursively find a Variable using that function
                    var = _find_var(name)
                    if var is not None:
                        return var
                # Get the function corresponding to the name from the mapping_str_func dictionary
                func = VariableFactory.mapping_str_func[name]
                var = func(
                    value, graph, tracker
                )  # Call the function to create a Variable from the value
                if var is not None:
                    return var

        var = _find_var()
        if var is None:
            var = VariableFactory.default_from_value(
                value, graph, tracker
            )  # If a Variable could not be found using the registered functions, use the default function to create a new Variable
        var.debug_name = debug_name
        return var


class VariableBase:
    """
    VariableBase is a basic concept and each symbols in VM stack is regarded as
    an Variable Object in symblic tracing process.

    There are two key data structures during Python runtime:
    PyFrameObject, which provides the instance for function logical lock usage,
    and PyCodeObject, which provides the bytecode for the corresponding function.
    With these data, the Python virtual machine executes the bytecode sequentially on a stack to complete function logic.

    Args:
        tracker(Tracker): The Tracker object that tracks the information of this variable.

    Note:
        We should push an object of a subclass of VariableBase instead of an object of VariableBase onto the VM stack.
        It serves as an abstract class and should not be instantiated directly.
    """

    tracker: Tracker  # An attribute to store the Tracker object associated with the variable
    name_generator = NameGenerator(
        "object_"
    )  # A class-level attribute to generate names for new variables

    def __init__(self, tracker: Tracker):
        self.tracker = tracker
        self.id = VariableBase.name_generator.next()
        self._debug_name: str | None = None

    @property
    def main_info(self) -> dict[str, Any]:
        """
        Property method to return a dictionary of main information about the variable

        Returns:
            main_info: Main information of the variable.
        """
        return {}

    @property
    def debug_info(self) -> dict[str, Any]:
        """
        Property method to return a dictionary of debug information about the variable
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
        if self._debug_name is not None:
            # Return the self._debug_name cache if it is not None.
            return self._debug_name
        inputs = self.tracker.inputs
        if isinstance(self.tracker, GetItemTracker):
            self._debug_name = (
                f"{self.tracker.container.debug_name}[{self.tracker.key}]"
            )
        elif isinstance(self.tracker, GetAttrTracker):
            self._debug_name = (
                f"{self.tracker.obj.debug_name}.{self.tracker.attr}"
            )
        elif len(inputs) == 0:
            self._debug_name = "tmp_var"
        else:  # len(inputs) >= 0
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

        frame_value_tracer = (
            self.tracker.trace_value_from_frame()
        )  # Get a ValueTracer object from the Tracker object associated with the variable
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
        """
        Abstract method to get the value of the variable
        """
        raise NotImplementedError()

    def get_type(self):
        """
        Method to get the type of the variable's value
        """
        return type(self.get_value())

    def reconstruct(self, codegen: PyCodeGen):
        if (
            not isinstance(self.tracker, DummyTracker)
            and self.tracker.is_traceable()
        ):
            self.tracker.gen_instructions(codegen)
        else:
            self.graph.add_global_guarded_variable(self)
            self._reconstruct(codegen)

    def _reconstruct(self, codegen: PyCodeGen):
        """
        Abstract method to construct an opcode and append it into codegen.instructions
        """
        raise NotImplementException()

    def flatten_items(self) -> list[VariableBase]:
        """
        Recursively flatten the items in this container variable to a list of Variable objects.

        Returns:
            list[VariableBase]: Flattened items of a container variable.
        """
        from .container import ContainerVariable

        if not isinstance(self, ContainerVariable):
            return [self]
        flattened_items = []
        for item in self.get_items():
            flattened_items.extend(item.flatten_items())
        return flattened_items

    def get_inputs(self) -> list[VariableBase]:
        """
        This method is used to get the inputs for the current variable.

        Returns:
            list[VariableBase]: Inputs for the current variable.
        """
        return self.tracker.inputs

    def get_traceable_inputs(self) -> list[VariableBase]:
        """
        This method is used to get the traceable inputs for the current variable.

        Returns:
            list[VariableBase]: Traceable inputs for the current variable.
        """
        return list(
            filter(lambda x: x.tracker.is_traceable(), self.tracker.inputs)
        )

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
        if inspect.ismethod(attr) or (
            hasattr(attr, "__self__")
            and inspect.ismethoddescriptor(
                getattr(attr.__self__.__class__, name, None)
            )
        ):
            from .callable import MethodVariable

            fn = None
            if inspect.ismethoddescriptor(
                getattr(attr.__self__.__class__, name, None)
            ):
                class_var = VariableFactory.from_value(
                    self.get_type(),
                    self.graph,
                    GetAttrTracker(self, "__class__"),
                )
                fn = VariableFactory.from_value(
                    getattr(attr.__self__.__class__, name),
                    self.graph,
                    GetAttrTracker(class_var, name),
                )
            return MethodVariable.wrap_method(
                value=attr,
                instance=self,
                fn=fn,
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
        output = fn_var(self, item)
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
        # if __call__ is a method, we should add self to arguments.
        if inspect.ismethod(self.get_value().__call__):
            args = (self,) + args
        unbound_method = get_unbound_method(self.get_value(), '__call__')
        if hasattr(unbound_method, "__code__"):
            fn_var = UserDefinedFunctionVariable(
                unbound_method,
                self.graph,
                GetAttrTracker(class_var, '__call__'),
            )
        else:
            fn_var = BuiltinVariable(
                self.value,
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
    ) -> VariableBase | None:
        """
        Create a new variable from a given value, or return None if the value cannot be converted to a variable.
        Args:
            value (Any): The value to create a variable from.
            graph (FunctionGraph | None): The graph in which the variable will be used.
            tracker (Tracker): The variable tracker to put the new variable in if created.

        Returns:
            VariableBase | None: A new variable if one can be created from the given value, or None if the value cannot be converted to a variable.
        """
        if isinstance(value, VariableBase):
            return value
        return None
