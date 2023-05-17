from __future__ import annotations

import paddle

from .infer_meta import MetaInfo
from .symbolic.statement_ir import Symbol
from .utils import NameGenerator, Singleton, log, no_eval_frame


# global variables
@Singleton
class ProxyTensorContext:
    def __init__(self):
        self.reset()

    def reset(self):
        self.runtime_name_to_proxy_tensor: dict[str, ProxyTensor] = {}
        self.runtime_proxy_tensor_to_name: dict[int, str] = {}
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

    def bind_name_to_proxy_tensor(self, name, proxy_tensor):
        self.runtime_name_to_proxy_tensor[name] = proxy_tensor
        self.runtime_proxy_tensor_to_name[id(proxy_tensor)] = name

    def clear_proxy_tensor_by_name(self, name):
        log(3, f"[GC] trying to GC {name}\n")
        proxy_tensor = self.runtime_name_to_proxy_tensor[name]
        proxy_tensor_id = id(proxy_tensor)
        has_value = proxy_tensor.value() is not None
        eager_tensor_id = id(proxy_tensor.value())

        del self.runtime_name_to_proxy_tensor[name]
        del self.runtime_proxy_tensor_to_name[proxy_tensor_id]
        if has_value and eager_tensor_id in self.tensor_to_proxy_tensor:
            del self.tensor_to_proxy_tensor[eager_tensor_id]
        log(3, f"[GC] {name} GCed\n")

    def get_runtime(self):
        return self.runtime_name_to_proxy_tensor


class ProxyTensor:
    def __init__(self, name, meta):
        self.name: str = name
        self.meta: MetaInfo = meta
        self.value_: paddle.Tensor = None
        ProxyTensorContext().bind_name_to_proxy_tensor(name, self)

    @property
    def shape(self):
        # TODO(xiongkun) consider dynamic shape.
        return self.meta.shape

    @property
    def ndim(self):
        return len(self.meta)

    def set_value(self, value):
        """
        value is a eager tensor.
        when a proxytensor have value, it means it can be evaluated outer to_static.
        """
        self.value_ = value

    def clear_value(self):
        self.value_ = None

    def value(self):
        return self.value_


@no_eval_frame
def convert_to_meta(inputs):
    def func(x):
        if isinstance(x, ProxyTensor):
            return x.meta
        return x

    return paddle.utils.map_structure(func, inputs)


@no_eval_frame
def convert_to_symbol(inputs):
    def func(x):
        if isinstance(x, ProxyTensor):
            return Symbol(x.name)
        return x

    pack_inputs = [inputs]
    ret = paddle.utils.map_structure(func, pack_inputs)
    return ret[0]


@no_eval_frame
def convert_arguments(inputs):
    # TODO (xionkgun): consider the following case:
    # >>> x = [tensor1, tensor2, tensor3]
    # >>> paddle.stack(x)
    # we should convert tensor to proxy tensor here.
    def func(x):
        if isinstance(x, paddle.Tensor):
            return ProxyTensorContext().from_tensor(x)
        return x

    return paddle.utils.map_structure(func, inputs)
