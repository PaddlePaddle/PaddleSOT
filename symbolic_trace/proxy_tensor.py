from __future__ import annotations

import paddle

from .infer_meta import MetaInfo
from .utils import NameGenerator, Singleton


# global variables
@Singleton
class ProxyTensorContext:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tensor_to_proxy_tensor: dict[int, ProxyTensor] = {}
        self.var_name_generator = NameGenerator("var_")

    def new_varname(self):
        return self.var_name_generator.next()

    def from_tensor(self, tensor) -> ProxyTensor:
        # TODO: don't have the same name.
        if self.tensor_to_proxy_tensor.get(id(tensor), None) is not None:
            return self.tensor_to_proxy_tensor[id(tensor)]

        # TODO(id may have collision)
        name = self.new_varname()
        proxy_tensor = ProxyTensor(name, MetaInfo.from_tensor(tensor))
        self.tensor_to_proxy_tensor[id(tensor)] = proxy_tensor
        proxy_tensor.set_value(tensor)
        return proxy_tensor


class ProxyTensor:
    def __init__(self, name, meta):
        self.name: str = name
        self.meta: MetaInfo = meta
        self.value_: paddle.Tensor = None

    def set_value(self, value):
        """
        value is a eager tensor.
        when a proxytensor have value, it means it can be evaluated outer to_static.
        """
        self.value_ = value

    def value(self):
        return self.value_
