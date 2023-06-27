from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from typing_extensions import Concatenate, ParamSpec, TypeAlias

    P = ParamSpec("P")
    R = TypeVar("R")

    DataGetter: TypeAlias = Callable[[Any, Any], Any]


class Mutation:
    ...


class MutationNew(Mutation):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"MutationNew({self.name}, {self.value})"


class MutationSet(Mutation):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"MutationSet({self.name}, {self.value})"


class MutationDel(Mutation):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"MutationDel({self.name})"


def record_mutation(
    mutation_fn: Callable[Concatenate[MutableDictLikeData, P], Mutation]
) -> Callable[Concatenate[MutableDictLikeData, P], None]:
    def wrapper(self, *args: P.args, **kwargs: P.kwargs):
        mutation = mutation_fn(self, *args, **kwargs)
        self.records.append(mutation)

    return wrapper


class MutableData:
    """
    An intermediate data structure between data and variable, it records all the mutations.
    """

    class Empty:
        def __repr__(self):
            return "Empty()"

    def __init__(self, data: Any, getter: DataGetter):
        self.original_data = data
        self.getter = getter
        self.records = []

    def is_empty(self, value):
        return isinstance(value, MutableData.Empty)

    @property
    def version(self):
        return len(self.records)

    @property
    def has_changed(self):
        return self.version != 0

    def get(self, key):
        raise NotImplementedError()

    def set(self, key, value):
        raise NotImplementedError()


class MutableDictLikeData(MutableData):
    def __init__(self, data: Any, getter: DataGetter):
        super().__init__(data, getter)
        self.read_cache = {}

    def clear(self):
        self.read_cache.clear()

    def get(self, key):
        write_cache = self.reproduce(self.version)
        if key not in write_cache:
            self.read_cache[key] = self.getter(self.original_data, key)
        return self.reproduce(self.version)[key]

    def get_all(self):
        original_keys = list(self.original_data.keys())
        for mutation in self.records:
            if isinstance(mutation, MutationNew):
                original_keys.append(mutation.name)
            elif isinstance(mutation, MutationDel):
                original_keys.remove(mutation.name)
        return {key: self.get(key) for key in original_keys}

    @record_mutation
    def set(self, key, value) -> Mutation:
        is_new = False
        if self.is_empty(self.get(key)):
            is_new = True
        return (
            MutationSet(key, value) if not is_new else MutationNew(key, value)
        )

    @record_mutation
    def delete(self, key):
        return MutationDel(key)

    def apply(self, mutation: Mutation, write_cache: dict[str, Any]):
        if isinstance(mutation, MutationNew):
            write_cache[mutation.name] = mutation.value
        elif isinstance(mutation, MutationSet):
            write_cache[mutation.name] = mutation.value
        elif isinstance(mutation, MutationDel):
            write_cache[mutation.name] = MutableData.Empty()
        else:
            raise ValueError(f"Unknown mutation type {mutation}")

    def reproduce(self, version: int | None = None):
        if version is None:
            version = self.version
        write_cache = self.read_cache.copy()
        for mutation in self.records[:version]:
            self.apply(mutation, write_cache)
        return write_cache

    def rollback(self, version: int):
        assert version <= self.version
        self.records[:] = self.records[:version]
