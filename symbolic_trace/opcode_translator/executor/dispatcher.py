from __future__ import annotations

from typing import Any


class Handler:
    def __init__(self, *types: type[Any]):
        self.types = types

    def match_types(self, *args: *Any) -> bool:
        return all(
            isinstance(arg, type_) for arg, type_ in zip(args, self.types)
        )
