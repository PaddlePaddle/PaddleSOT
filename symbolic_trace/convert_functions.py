from .utils import no_eval_frame, log, is_fallback_api, is_paddle_api
from .proxy_tensor import ProxyTensorContext, callable_wrapper
import paddle

log_level = 10


def convert_callable(func):
    if is_paddle_api(func) or is_fallback_api(func): 
        return callable_wrapper(func)
    return func

def convert_tensor(tensor):
    return ProxyTensorContext().from_tensor(tensor)

@no_eval_frame
def convert_function(obj):
    if callable(obj):
        obj = convert_callable(obj)
    elif isinstance(obj, paddle.Tensor):
        obj = convert_tensor(obj)
    return obj
