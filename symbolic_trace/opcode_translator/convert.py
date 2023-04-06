import paddle
from ..proxy_tensor import paddle_api_wrapper, ProxyTensorContext
from ..utils import log, no_eval_frame


def convert_one(obj):
    # 1. use contextmanager to change frame callback will lead to err
    # 2. can not use decorator 'no_eval_frame' here, which will lead to infinite loop
    if obj is paddle.fluid.core.set_eval_frame:
        return obj
    old_cb = paddle.fluid.core.set_eval_frame(None)

    log_level = 10
    log(log_level, "[convert] " + f"target: {obj}    ")
    if callable(obj):
        log(log_level, "found a callable object\n")
        obj = convert_callable(obj)
    elif isinstance(obj, paddle.Tensor):
        log(log_level, "found a tensor\n")
        obj = convert_tensor(obj)
    else:
        log(log_level, "nothing happend\n")
    
    paddle.fluid.core.set_eval_frame(old_cb)
    return obj

def convert_multi(args):
    old_cb = paddle.fluid.core.set_eval_frame(None)
    retval = []
    for obj in args:
        retval.append(convert_one(obj))
    retval = tuple(retval)
    paddle.fluid.core.set_eval_frame(old_cb)
    return retval
  
def convert_callable(func):
    if isinstance(func, type):
        return func
    return paddle_api_wrapper(func)

def convert_tensor(tensor):
    return ProxyTensorContext().from_tensor(tensor)

@no_eval_frame
def convert_return(retval):
    print("convert_return")
    return retval