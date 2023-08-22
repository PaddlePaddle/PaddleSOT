from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from .mutable_data import MutableData
from .variables import VariableBase

if TYPE_CHECKING:
    from .mutable_data import DataGetter
    from .pycode_generator import PyCodeGen

    MutableDataT = TypeVar("MutableDataT", bound=MutableData)


class SideEffectsState(NamedTuple):
    data_id_to_proxy: dict[int, MutableData]
    variables: list[VariableBase]
    proxy_versions: list[int]


class SideEffects:
    def __init__(self):
        self.data_id_to_proxy: dict[int, MutableData] = {}
        self.variables: list[VariableBase] = []

    def record_variable(self, variable: VariableBase):
        if variable not in self.variables:
            self.variables.append(variable)

    def get_proxy(
        self,
        proxy_type: type[MutableDataT],
        data: Any,
        getter: DataGetter,
    ) -> MutableDataT:
        data_id = id(data)
        if data_id not in self.data_id_to_proxy:
            self.data_id_to_proxy[data_id] = proxy_type(data, getter)
        return self.data_id_to_proxy[data_id]  # type: ignore

    def get_state(self):
        return SideEffectsState(
            self.data_id_to_proxy.copy(),
            self.variables.copy(),
            [proxy.version for proxy in self.data_id_to_proxy.values()],
        )

    def restore_state(self, state: SideEffectsState):
        self.data_id_to_proxy = state.data_id_to_proxy
        self.variables = state.variables
        for proxy, version in zip(
            self.data_id_to_proxy.values(), state.proxy_versions
        ):
            proxy.rollback(version)


class SideEffectRestorer:
    def __init__(self):
        ...

    def pre_gen(self, codegen: PyCodeGen):
        raise NotImplementedError()

    def post_gen(self, codegen: PyCodeGen):
        raise NotImplementedError()


class DictSideEffectRestorer(SideEffectRestorer):
    """
    old_dict.clear()
    old_dict.update(new_dict)
    """

    def __init__(self, var: VariableBase):
        super().__init__()
        self.var = var

    def pre_gen(self, codegen: PyCodeGen):
        # Reference to the original dict.
        # load old_dict.update and new_dict to stack.
        self.var.reconstruct(codegen)
        codegen.gen_load_method("update")
        # Generate dict by each key-value pair.
        self.var.reconstruct(codegen, use_tracker=False)
        # load old_dict.clear to stack.
        self.var.reconstruct(codegen)
        codegen.gen_load_method("clear")

    def post_gen(self, codegen: PyCodeGen):
        # Call methods to apply side effects.
        codegen.gen_call_method(0)  # call clear
        codegen.gen_pop_top()
        codegen.gen_call_method(1)  # call update
        codegen.gen_pop_top()


class ListSideEffectRestorer(SideEffectRestorer):
    """
    old_list[:] = new_list
    """

    def __init__(self, var: VariableBase):
        super().__init__()
        self.var = var

    def pre_gen(self, codegen: PyCodeGen):
        # Reference to the original list.
        # load new_list to stack.
        self.var.reconstruct(codegen, use_tracker=False)
        # load old_list[:] to stack.
        self.var.reconstruct(codegen)
        codegen.gen_load_const(None)
        codegen.gen_load_const(None)
        codegen.gen_build_slice(2)

    def post_gen(self, codegen: PyCodeGen):
        # Call STROE_SUBSCR to apply side effects.
        codegen.gen_store_subscr()


class GlobalSetSideEffectRestorer(SideEffectRestorer):
    """
    global_var = new_value
    """

    def __init__(self, name: str, var: VariableBase):
        super().__init__()
        self.name = name
        self.var = var

    def pre_gen(self, codegen: PyCodeGen):
        self.var.reconstruct(codegen)

    def post_gen(self, codegen: PyCodeGen):
        codegen.gen_store_global(self.name)


class GlobalDelSideEffectRestorer(SideEffectRestorer):
    """
    del global_var
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def pre_gen(self, codegen: PyCodeGen):
        # do nothing
        ...

    def post_gen(self, codegen: PyCodeGen):
        codegen.gen_delete_global(self.name)
