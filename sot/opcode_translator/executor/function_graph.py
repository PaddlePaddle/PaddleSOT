# This file is specifically used to handle the problem
# of generating a Graph from a linear function call.

from __future__ import annotations

import builtins
from collections import namedtuple
from copy import deepcopy
from functools import cached_property
from typing import Any, Callable

from ...infer_meta import InferMetaCache, LayerInferMetaCache, MetaInfo
from ...symbolic.statement_ir import Symbol
from ...symbolic.symbolic_context import SymbolicTraceContext
from ...utils import (
    EventGuard,
    NameGenerator,
    OrderedSet,
    event_register,
    inner_error_default_handler,
    is_paddle_api,
    log,
    log_do,
    map_if,
    show_trackers,
)
from .guard import Guard, StringifyExpression, make_guard
from .pycode_generator import PyCodeGen
from .side_effects import SideEffects
from .tracker import BuiltinTracker, ConstTracker, DummyTracker
from .variables import (
    ContainerVariable,
    DictVariable,
    DummyVariable,
    ListVariable,
    PaddleLayerVariable,
    TensorVariable,
    VariableBase,
    VariableFactory,
    find_traceable_vars,
    map_variables,
)


def convert_to_meta(inputs: Any):
    """
    Convert the input variables to meta if it is TensorVariable.
    """

    def func(x):
        if isinstance(x, TensorVariable):
            return x.meta
        return x.get_py_value()

    return map_variables(func, inputs)


def convert_to_symbol(inputs: Any):
    """
    Convert the input variables to symbol if it can be symbolic.
    """

    def func(x):
        if isinstance(x, (TensorVariable, PaddleLayerVariable)):
            return x.get_symbol()
        return x.get_py_value()

    return map_variables(func, inputs)


class FunctionGraph:
    """
    A Graph representation corresponding to each FunctionFrame
    The input binding diagram containing the current call represents three parts of output settings,
    This Graph can be compiled as a f_locals dependency function which produce the same outputs.
    """

    OUT_VAR_PREFIX = "___SIR_out_"
    Memo = namedtuple(
        "function_graph_memo",
        [
            'inner_out',
            'input_variables',
            "stmt_ir",
            "global_guards",
            "side_effects_state",
            "print_variables",
        ],
    )

    def __init__(self, frame, **kwargs):
        self.sir_ctx = SymbolicTraceContext()
        self.inner_out = set()
        self.input_variables = []  # Store variables required within a function
        self.pycode_gen = PyCodeGen(frame, disable_eval_frame=True)
        self.side_effects = SideEffects()
        self.py_frame = frame
        self._global_guarded_variables: OrderedSet[VariableBase] = OrderedSet()
        self._print_variables = []
        self.build_strategy = kwargs.get('build_strategy', None)

    @cached_property
    def _builtins(self):
        builtins_ = {}
        # prepare builtins
        for name, value in builtins.__dict__.items():
            builtins_[name] = VariableFactory.from_value(
                value, self, BuiltinTracker(name), debug_name=name
            )
        return builtins_

    def add_print_variables(self, variable):
        """
        Used to support psdb_print
        """
        self._print_variables.append(variable)

    def need_add_input(self, var):
        """
        Determine if it is the input of graph.

        Args:
            var: The input variable.

        """
        if var.id in self.inner_out:
            return False
        for v in self.input_variables:
            if v.id == var.id:
                return False
        return True

    def save_memo(self) -> FunctionGraph.Memo:
        """
        Save the state of the current FunctionGraph, for future state recovery, it is used for state recovery during inline call error reporting

        NOTE:
            Why don't use __deepcopy__, because memo is not a deepcopy, i.e inner_out is only a shallow copy, SIR is a deepcopy.
        """
        with EventGuard(f"Save SIR Checkpoint: len({len(self.sir_ctx.TOS)})"):
            saved_stmt_ir = deepcopy(self.sir_ctx.TOS)
            return FunctionGraph.Memo(
                inner_out=set(self.inner_out),
                input_variables=list(self.input_variables),
                stmt_ir=saved_stmt_ir,
                global_guards=OrderedSet(self._global_guarded_variables),
                side_effects_state=self.side_effects.get_state(),
                print_variables=list(self._print_variables),
            )

    def restore_memo(self, memo: FunctionGraph.Memo):
        """
        Restore the state of graph to memo.

        Args:
            memo: Previously recorded memo

        """
        self.inner_out = memo.inner_out
        self.input_variables = memo.input_variables
        self.sir_ctx.replace_TOS(memo.stmt_ir)
        self._global_guarded_variables = memo.global_guards
        self.side_effects.restore_state(memo.side_effects_state)
        self._print_variables = memo.print_variables

    def collect_input_variables(self, inputs: list[VariableBase]):
        """
        Variables required within the method

        Args:
            inputs: Required VariableBase
        """
        for inp in inputs:
            if isinstance(inp, ContainerVariable):
                self.collect_input_variables(inp.get_items())
            if isinstance(inp, VariableBase) and self.need_add_input(inp):
                self.input_variables.append(inp)

    @property
    @event_register("guard_fn")
    def guard_fn(self) -> Guard:
        guards = [
            variable.make_stringify_guard()
            for variable in find_traceable_vars(
                self.input_variables + list(self._global_guarded_variables)
            )
            if not isinstance(variable.tracker, (DummyTracker, ConstTracker))
        ]

        guards = list(set(guards))

        for guard in guards:
            assert isinstance(
                guard, StringifyExpression
            ), "guard must be StringifyExpression."

        return make_guard(guards)

    def start_compile_with_name_store(self, ret_vars, to_store_vars):
        class VariableLoader:
            def __init__(self, index_for_load, pycode_gen):
                self._index_for_load = index_for_load
                self._pycode_gen = pycode_gen

            def load(self, var):
                if isinstance(var, DummyVariable):
                    var.reconstruct(self._pycode_gen)
                    return
                self._pycode_gen.gen_load_fast(self._index_for_load[var.id])

        # var_id -> local_name mapping
        index_for_load = {}
        to_store_vars = list(
            filter(lambda x: not isinstance(x, DummyVariable), to_store_vars)
        )
        self.start_compile(*(ret_vars + to_store_vars))
        name_gen = NameGenerator("__start_compile_saved_")
        for var in to_store_vars:
            index_for_load[var.id] = name_gen.next()

            def _log_fn():
                print(
                    f"[StartCompile] saved var: {index_for_load[var.id]} = ",
                    var,
                )

            log_do(4, _log_fn)

        for var in to_store_vars[::-1]:
            self.pycode_gen.gen_store_fast(index_for_load[var.id])
        return VariableLoader(index_for_load, self.pycode_gen)

    @event_register("start_compile")
    def start_compile(self, *ret_vars: VariableBase):
        """
        Generate bytecode based on the information collected by the simulation execution.

        This consists of the following steps:
        - Compile the FunctionGraph into a dy2st StaticFunction and load it in the generated bytecode
        - Load the group network input
        - Calling the generated dy2st StaticFunction
        - Restore the side effects
        - Restore the output
        - Return the top of the stack
        """
        from ..breakpoint import BreakpointManager

        BreakpointManager().on_event("start_compile")

        ret_items = [
            ret_item
            for ret_var in ret_vars
            for ret_item in ret_var.flatten_items()
        ]

        tensor_items = self._find_tensor_outputs(ret_items)
        compiled_fn, statment_ir = self.sir_ctx.compile_fn(
            [Symbol(tensor_var.var_name) for tensor_var in tensor_items],
            self.build_strategy,
        )
        input_names = statment_ir.inputs
        compiled_fn_name = f"__compiled_fn_{statment_ir.name}"
        # prepare function and inputs
        self.pycode_gen.gen_load_object(compiled_fn, compiled_fn_name)
        for name in input_names:
            found = False
            for variable in self.input_variables:
                if (
                    isinstance(variable, (TensorVariable, PaddleLayerVariable))
                    and variable.get_symbol().name == name
                ):
                    variable.tracker.gen_instructions(self.pycode_gen)
                    found = True
                    break
            assert found, f"can't find input {name} in SIR."
        # Pack all args into a tuple, because we don't support *args now.
        self.pycode_gen.gen_build_tuple(count=len(input_names))
        # call the compiled_fn
        self.pycode_gen.gen_call_function(argc=1)

        # Store outputs to f_locals
        self.pycode_gen.gen_unpack_sequence(count=len(tensor_items))
        for tensor_var in tensor_items:
            self.pycode_gen.gen_store_fast(tensor_var.out_var_name)
        # restore the outputs.
        for ret_var in ret_vars:
            ret_var.reconstruct(self.pycode_gen)

        # deal side effect
        self.restore_print_stmts(self._print_variables)
        self.restore_side_effects(self.side_effects.variables)

        tracker_output_path = show_trackers()
        if tracker_output_path:
            from .tracker_viewer import view_tracker

            view_tracker(list(ret_vars), tracker_output_path, format="png")

    def call_paddle_api(
        self,
        func: Callable[..., Any],
        *args: VariableBase,
        **kwargs: VariableBase,
    ):
        """
        Record Paddle Networking API to SIR

        Args:
            func: paddle api
        """
        assert is_paddle_api(func)
        # not fallback api, start symbolic trace.
        # TODO(xiokgun): may have python buildin object inside metas.
        # TODO(xiokgun): 4 kinds of python arguments. support it !!
        log(3, f"call paddle.api : {func.__name__}", "\n")

        def message_handler(*args, **kwargs):
            return f"Call paddle_api error: {func.__name__}, may be not a operator api ?"

        return inner_error_default_handler(self.symbolic_call, message_handler)(
            InferMetaCache(), self.sir_ctx.call_API, func, *args, **kwargs
        )

    def call_tensor_method(
        self, method_name: str, *args: VariableBase, **kwargs
    ):
        """
        call tensor method, start symbolic trace.

        Args:
            method_name: tensor method name
        """

        def message_handler(*args, **kwargs):
            return f"Call tensor_method error: Tensor.{method_name}, may be not a valid operator api ?"

        return inner_error_default_handler(self.symbolic_call, message_handler)(
            InferMetaCache(),
            self.sir_ctx.call_METHOD,
            method_name,
            *args,
            **kwargs,
        )

    def call_layer(
        self,
        layer: PaddleLayerVariable,
        *args: VariableBase,
        **kwargs: VariableBase,
    ):
        """
        call paddle layer, start symbolic trace.

        Args:
            layer: paddle layer
        """

        def infer_meta_fn(layer, *metas, **kwmetas):
            metas = metas[1:]
            metas = LayerInferMetaCache()(layer.value, *metas, **kwmetas)
            return metas

        def compute_fn(layer, inputs, outputs):
            inputs = (layer.get_symbol(), *inputs)
            inputs = inputs[1:]
            self.sir_ctx.call_LAYER(
                layer.value.__class__.__name__,
                inputs=inputs,
                outputs=outputs,
            )

        def message_handler(*args, **kwargs):
            return f"Call paddle layer error: {layer}, may be not a valid paddle layer ?"

        return inner_error_default_handler(self.symbolic_call, message_handler)(
            infer_meta_fn, compute_fn, layer, *[layer, *args], **kwargs
        )

    def symbolic_call(self, infer_meta_fn, compute_fn, func, *args, **kwargs):
        """
        Using infer_meta_fn and compute_fn convert func to symbolic function.

        Args:
            infer_meta_fn: function for infer meta, (func, metas, kwmetas) -> output_metas
            compute_fn   : function for sir compile, (func, input_symbols, outputs_symbols) -> None
            func         : symbolic function
        """
        self.collect_input_variables(list(args))
        self.collect_input_variables(list(kwargs.values()))
        metas = convert_to_meta(args)
        kwmetas = convert_to_meta(kwargs)

        with EventGuard("infer_meta"):
            out_metas = infer_meta_fn(func, *metas, **kwmetas)
        inputs_symbols = (
            convert_to_symbol(args),
            convert_to_symbol(kwargs),
        )
        log(3, f"         inputs : {inputs_symbols}", "\n")
        outputs = map_if(
            out_metas,
            pred=lambda x: isinstance(x, MetaInfo),
            true_fn=lambda x: TensorVariable(
                x,
                self,
                tracker=DummyTracker(list(args) + list(kwargs.values())),
            ),
            false_fn=lambda x: x,
        )
        if outputs is not None:
            compute_fn(
                func, inputs_symbols, convert_to_symbol(outputs)
            )  # symbolic only contain symbols.
            self._put_inner(outputs)
            return VariableFactory.from_value(
                outputs, self, DummyTracker(list(args) + list(kwargs.values()))
            )
        else:
            return None

    def _put_inner(self, var: VariableBase):
        """
        put inner variable to inner_out
        """
        map_if(
            var,
            pred=lambda x: isinstance(x, VariableBase),
            true_fn=lambda x: self.inner_out.add(x.id),
            false_fn=lambda x: None,
        )

    def add_global_guarded_variable(self, variable: VariableBase):
        """
        Add variable to global guarded variable
        """
        self._global_guarded_variables.add(variable)

    def remove_global_guarded_variable(self, variable: VariableBase):
        """
        Remove variable to global guarded variable
        """
        if variable in self._global_guarded_variables:
            self._global_guarded_variables.remove(variable)

    def _find_tensor_outputs(
        self, outputs: list[VariableBase]
    ) -> OrderedSet[TensorVariable]:
        """
        Return all TensorVariable. find TensorVariables participating in networking from the output Variables

        Args:
            outputs: output variables
        """
        output_tensors: OrderedSet[TensorVariable] = OrderedSet()
        # Find Tensor Variables from outputs.
        for output in outputs:
            if isinstance(output.tracker, DummyTracker):
                if isinstance(output, TensorVariable):
                    output_tensors.add(output)
                else:
                    # Guard output that can not be traced.
                    self.add_global_guarded_variable(output)
        # Find Tensor Variables from side effects Variables.
        for side_effect_var in self.side_effects.variables:
            if side_effect_var.proxy.has_changed:
                for var in side_effect_var.flatten_items():
                    if isinstance(var.tracker, DummyTracker) and isinstance(
                        var, TensorVariable
                    ):
                        output_tensors.add(var)
        # Find Tensor in print_stmts
        for print_stmt in self._print_variables:
            for var in print_stmt.flatten_items():
                if isinstance(var.tracker, DummyTracker) and isinstance(
                    var, TensorVariable
                ):
                    output_tensors.add(var)
        return output_tensors

    def restore_print_stmts(self, variables: list[VariableBase]):
        for var in variables:
            var.reconstruct(
                self.pycode_gen,
                use_tracker=False,
                add_to_global_guarded_vars=False,
            )

    def restore_side_effects(self, variables: list[VariableBase]):
        """
        Generate side effect recovery code for variables with side effects

        Args:
            variables: Variables that may have side effects.
        """

        if not variables:
            return

        var = variables[0]
        # skip inner variables
        if not var.tracker.is_traceable():
            self.restore_side_effects(variables[1:])
            return
        if isinstance(var, DictVariable):
            # old_dict.clear()
            # old_dict.update(new_dict)

            # Reference to the original dict.
            # load old_dict.update and new_dict to stack.
            var.reconstruct(self.pycode_gen)
            self.pycode_gen.gen_load_method("update")
            # Generate dict by each key-value pair.
            var.reconstruct(self.pycode_gen, use_tracker=False)
            # load old_dict.clear to stack.
            var.reconstruct(self.pycode_gen)
            self.pycode_gen.gen_load_method("clear")

            # Generate side effects of other variables.
            self.restore_side_effects(variables[1:])

            # Call methods to apply side effects.
            self.pycode_gen.gen_call_method(0)  # call clear
            self.pycode_gen.gen_pop_top()
            self.pycode_gen.gen_call_method(1)  # call update
            self.pycode_gen.gen_pop_top()
        elif isinstance(var, ListVariable):
            # old_list[:] = new_list

            # Reference to the original list.
            # load new_list to stack.
            var.reconstruct(self.pycode_gen, use_tracker=False)
            # load old_list[:] to stack.
            var.reconstruct(self.pycode_gen)
            self.pycode_gen.gen_load_const(None)
            self.pycode_gen.gen_load_const(None)
            self.pycode_gen.gen_build_slice(2)

            # Generate side effects of other variables.
            self.restore_side_effects(variables[1:])

            # Call STROE_SUBSCR to apply side effects.
            self.pycode_gen.gen_store_subscr()
