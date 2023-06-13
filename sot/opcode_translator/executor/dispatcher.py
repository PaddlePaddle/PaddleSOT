from __future__ import annotations

from typing import Any, Callable

from sot.utils import Singleton

from .variables import ConstVariable, DictVariable, VariableBase


class Pattern:
    def __init__(self, *types: type[Any], **kwtypes: type[Any]):
        self.types = types
        self.kwtypes = kwtypes

    def match_inputs(self, *args: Any, **kwargs: Any) -> bool:
        if len(args) != len(self.types):
            return False
        if any(name not in kwargs for name in self.kwtypes.keys()):
            return False
        return all(
            isinstance(arg, type_) for arg, type_ in zip(args, self.types)
        ) and all(
            isinstance(kwargs[name], type_)
            for name, type_ in self.kwtypes.items()
        )

    def __repr__(self) -> str:
        types_repr = ", ".join([type_.__name__ for type_ in self.types])
        kwtypes_repr = ", ".join(
            [f"{name}={type_.__name__}" for name, type_ in self.kwtypes.items()]
        )
        return f"Pattern({types_repr}, {kwtypes_repr})"


@Singleton
class Dispatcher:
    handlers: dict[Callable[..., Any], list[tuple[Pattern, Callable[..., Any]]]]

    def __init__(self):
        self.handlers = {}
        # dict
        self.register(
            dict.keys,
            (DictVariable,),
            {},
            lambda var: var.override_method_keys(),
        )
        self.register(
            dict.update,
            (DictVariable, DictVariable),
            {},
            lambda var, other: var.override_method_update(other),
        )
        # getattr
        # TODO(SigureMo): Unify these to a single function
        self.register(
            getattr,
            (VariableBase, str),
            {},
            lambda var, name: var.getattr(name),
        )
        self.register(
            getattr,
            (VariableBase, ConstVariable),
            {},
            lambda var, name: var.getattr(name.get_value()),
        )

    def register(
        self,
        fn: Callable[..., Any],
        types: tuple[type[Any], ...],
        kwtypes: dict[str, type[Any]],
        handler: Callable[..., Any],
    ):
        if fn not in self.handlers:
            self.handlers[fn] = []
        self.handlers[fn].append((Pattern(*types, **kwtypes), handler))

    def dispatch(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Callable[..., Any] | None:
        if fn not in self.handlers:
            return None
        for pattern, handler in self.handlers[fn]:
            if pattern.match_inputs(*args, **kwargs):
                return handler
        return None
