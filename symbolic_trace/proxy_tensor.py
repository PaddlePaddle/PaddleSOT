import paddle
from .symbolic_trace import SymbolicTraceContext

# global variables
runtime_name_to_eager_tensor = {}
runtime_eager_tensor_to_name = {}

class MetaInfo: 
    def __init__(self, shape, dtype, stop_gradient):
        self.shape = shape
        self.dtype = dtype
        self.stop_gradient = stop_gradient

    @staticmethod
    def from_tensor(tensor):
        return MetaInfo(tensor.shape, tensor.dtype, tensor.stop_gradient)

def clear_runtime_proxytensor():
    runtime_name_to_eager_tensor.clear()
    runtime_eager_tensor_to_name.clear()


class ProxyTensor:
    def __init__(self, name, meta):
        self.name = name
        self.meta = meta

    @staticmethod
    def from_tensor(tensor):
        if runtime_eager_tensor_to_name.get(id(tensor), None) is not None:
            return runtime_eager_tensor_to_name[id(tensor)]

        #TODO(id may have collision)
        name = SymbolicTraceContext().new_varname()
        runtime_name_to_eager_tensor[name] = tensor
        runtime_eager_tensor_to_name[id(tensor)] = name 
        return ProxyTensor(name, MetaInfo.from_tensor(tensor))

    def __add__(self, other):
        # later we will use variable shape inference to infer the shape of the output
        return paddle_api_wrapper(paddle.add)(self, other)

    def __radd__(self, other):
        # later we will use variable shape inference to infer the shape of the output
        return paddle_api_wrapper(paddle.add)(other, self)

    def __sub__(self, other):
        # later we will use variable shape inference to infer the shape of the output
        return paddle_api_wrapper(paddle.substract)(self, other)

    def __rsub__(self, other):
        # later we will use variable shape inference to infer the shape of the output
        return paddle_api_wrapper(paddle.substract)(other, self)

def infer_meta(func, *args):
    args = convert_to_meta_tensor(args)
    if func in [paddle.add, paddle.subtract]: 
        x_meta, y_meta = args
        if isinstance(x_meta, MetaInfo): 
            return MetaInfo(x_meta.shape, x_meta.dtype, x_meta.stop_gradient)
        else: 
            return MetaInfo(y_meta.shape, y_meta.dtype, y_meta.stop_gradient)

def convert_to_meta_tensor(inputs):
    def func(x):
        if isinstance(x, ProxyTensor): 
            return x.meta
        return x
    return paddle.utils.map_structure(func, inputs)
    

def convert_to_name(inputs):
    def func(x):
        if isinstance(x, ProxyTensor): 
            return x.name
        return x
    return paddle.utils.map_structure(func, inputs)


def convert_arguments(inputs):
    #TODO (xionkgun): consider the following case: 
    # >>> x = [tensor1, tensor2, tensor3]
    # >>> paddle.stack(x)
    # we should convert tensor to proxy tensor here.
    def func(x):
        if isinstance(x, paddle.Tensor):
            return ProxyTensor.from_tensor(x)
        return x
    return paddle.utils.map_structure(func, inputs)

def paddle_api_wrapper(func):
    # NOTICE(zhanfei): codes in this wrapper will be transformed
    # maybe move the logic to another function?
    def wrapper(*args): 
        old_cb = paddle.fluid.core.set_eval_frame(None)

        args = convert_arguments(args)
        # TODO(xiokgun): multi-output support.
        # TODO(xiokgun): may have python buildin object inside metas.
        # TODO(xiokgun): 4 kinds of python arguments. support it !!

        if func in [ paddle.add, paddle.subtract]:
            meta = infer_meta(func, *args)
            result = ProxyTensor(SymbolicTraceContext().new_varname(), meta)
            SymbolicTraceContext().call_API(func, inputs=convert_to_name(args), outputs=result.name) # symbolic only contain symbols.
            return result
        retval = func(*args)

        paddle.fluid.core.set_eval_frame(old_cb)
        return retval
    return wrapper
