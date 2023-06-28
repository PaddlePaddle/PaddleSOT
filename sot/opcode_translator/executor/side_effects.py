from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from .mutable_data import MutableData
from .variables import VariableBase

if TYPE_CHECKING:
    from .mutable_data import DataGetter

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
            self.data_id_to_proxy,
            self.variables,
            [proxy.version for proxy in self.data_id_to_proxy.values()],
        )

    def restore_state(self, state: SideEffectsState):
        self.data_id_to_proxy = state.data_id_to_proxy
        self.variables = state.variables
        for proxy, version in zip(
            self.data_id_to_proxy.values(), state.proxy_versions
        ):
            proxy.rollback(version)
