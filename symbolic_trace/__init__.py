from .trace import symbolic_trace
from .monkey_patch import patch_proxy_tensor
patch_proxy_tensor()

__all__ = [
    'symbolic_trace',
]
