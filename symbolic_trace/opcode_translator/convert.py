import paddle
from ..utils import log, no_eval_frame, Singleton

@Singleton
class Callbacks:
    def __init__(self):
        self.on_convert = None

    def set_on_convert(self, on_convert):
        self.on_convert = on_convert

    def has_callback(self):
        return self.on_convert is not None

def convert_one(obj):
    # 1. use contextmanager to change frame callback will lead to err
    # 2. can not use decorator 'no_eval_frame' here, which will lead to infinite loop
    if obj is paddle.fluid.core.set_eval_frame:
        return obj
    old_cb = paddle.fluid.core.set_eval_frame(None)

    log_level = 10
    log(log_level, "[convert] " + f"target: {obj}    ")
    if Callbacks().has_callback():
        obj = Callbacks().on_convert(obj)
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
  
@no_eval_frame
def convert_return(retval):
    return retval

