import paddle
from ..proxy_tensor import ProxyTensor, paddle_api_wrapper

CONVERT_SKIP_NAMES = (
    "convert_one", 
    "convert_multi",  
)

LOG_FLAG = True

def log(*args):
    if LOG_FLAG:
        print(*args, end="")

def convert_one(obj):
    # use contextmanager to change frame callback will lead to err
    old_cb = paddle.fluid.core.set_eval_frame(None)
    log(f"convert: {obj}    ")
    if callable(obj):
        log("found a callable object\n")
        return convert_callable(obj)
    if isinstance(obj, paddle.Tensor):
        log("found a tensor\n")
        return convert_tensor(obj)
    log("nothing happend\n")
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

def enable_log(with_log):
    convert.LOG_FLAG = with_log

