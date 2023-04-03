import paddle
from ..proxy_tensor import ProxyTensor, paddle_api_wrapper
from ..utils import log

CONVERT_SKIP_NAMES = (
    "convert_one", 
    "convert_multi",  
)

def convert_one(obj):
    # use contextmanager to change frame callback will lead to err
    old_cb = paddle.fluid.core.set_eval_frame(None)
    log(10, f"convert: {obj}    ")
    if callable(obj):
        log(10, "found a callable object\n")
        obj = convert_callable(obj)
    elif isinstance(obj, paddle.Tensor):
        log(10, "found a tensor\n")
        obj = convert_tensor(obj)
    log(10, "nothing happend\n")
    paddle.fluid.core.set_eval_frame(old_cb)
    return obj

def convert_multi(args):
    old_cb = paddle.fluid.core.set_eval_frame(None)
    retval = []
    for obj in args:
        retval.append(convert_one(obj))
    paddle.fluid.core.set_eval_frame(old_cb)
    return tuple(retval)
  
def convert_callable(func):
    if isinstance(func, type):
        return func
    return paddle_api_wrapper(func)

def convert_tensor(tensor):
    return ProxyTensor.from_tensor(tensor)

