from .proxy_tensor import ProxyTensor
from .trace import symbolic_trace
from .utils import paddle_tensor_method
from .utils.monkey_patch import do_monkey_patch, proxy_tensor_method_builder

do_monkey_patch(ProxyTensor, paddle_tensor_method, proxy_tensor_method_builder)

__all__ = [
    "symbolic_trace",
]
