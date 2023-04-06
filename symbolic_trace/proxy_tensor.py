import paddle
from .symbolic_trace import SymbolicTraceContext
from .statement_ir import Symbol
from .utils import Singleton
from .opcode_translator import black_name_list

# global variables
@Singleton
class ProxyTensorContext:
    def __init__(self):
        self.reset()

    def reset(self):
        self.runtime_name_to_proxy_tensor = {}
        self.runtime_proxy_tensor_to_name = {}
        self.tensor_to_proxy_tensor = {}

    def from_tensor(self, tensor):
        #TODO: don't have the same name.
        if self.tensor_to_proxy_tensor.get(id(tensor), None) is not None:
            return self.tensor_to_proxy_tensor[id(tensor)] 

        #TODO(id may have collision)
        name = SymbolicTraceContext().new_varname()
        proxy_tensor = ProxyTensor(name, MetaInfo.from_tensor(tensor))
        self.runtime_name_to_proxy_tensor[name] = proxy_tensor
        self.runtime_proxy_tensor_to_name[id(proxy_tensor)] = name 
        self.tensor_to_proxy_tensor[id(tensor)] = proxy_tensor
        proxy_tensor.set_value(tensor)
        return proxy_tensor

    def get_runtime(self):
        return self.runtime_name_to_proxy_tensor

class MetaInfo: 
    def __init__(self, shape, dtype, stop_gradient):
        self.shape = shape
        self.dtype = dtype
        self.stop_gradient = stop_gradient

    @staticmethod
    def from_tensor(tensor):
        return MetaInfo(tensor.shape, tensor.dtype, tensor.stop_gradient)


class ProxyTensor:
    def __init__(self, name, meta):
        self.name = name
        self.meta = meta
        self.value_ = None

    def set_value(self, value):
        """
        value is a eager tensor. 
        when a proxytensor have value, it means it can be evaluated outer to_static.
        """
        self.value_ = value

    def value(self):
        return self.value_

    def __add__(self, other):
        # later we will use variable shape inference to infer the shape of the output
        return self.call_method("__add__", self, other)

    def __radd__(self, other):
        # later we will use variable shape inference to infer the shape of the output
        return self.call_method("__radd__", self, other)

    def __sub__(self, other):
        # later we will use variable shape inference to infer the shape of the output
        return self.call_method("__sub__", self, other)

    def __rsub__(self, other):
        # later we will use variable shape inference to infer the shape of the output
        return self.call_method("__rsub__", self, other)

    def __bool__(self):
        # TODO: (too ugly, need to be refactored)
        old_cb = paddle.fluid.core.set_eval_frame(None)
        SymbolicTraceContext().start_compile(ProxyTensorContext().get_runtime())
        assert self.value() is not None
        paddle.fluid.core.set_eval_frame(old_cb)
        return bool(self.value())

    @staticmethod
    def call_method(method_name, *args):
        args = convert_arguments(args)
        if method_name in [ "__add__", "__radd__", "__sub__", "__rsub__" ]:
            meta = infer_meta(method_name, *args)
            result = ProxyTensor(SymbolicTraceContext().new_varname(), meta)
            SymbolicTraceContext().call_METHOD(method_name, inputs=convert_to_symbol(args), outputs=convert_to_symbol(result)) # symbolic only contain symbols.
            return result
        


def infer_meta(func, *args):
    args = convert_to_meta_tensor(args)
    if func in [paddle.add, paddle.subtract, "__add__", "__radd__", "__sub__", "__rsub__"]: 
        x_meta, y_meta = args
        if isinstance(x_meta, MetaInfo): 
            return MetaInfo(x_meta.shape, x_meta.dtype, x_meta.stop_gradient)
        else: 
            return MetaInfo(y_meta.shape, y_meta.dtype, y_meta.stop_gradient)
    elif func in [paddle.nn.functional.relu]:
        x_meta = args[0]
        return MetaInfo(x_meta.shape, x_meta.dtype, x_meta.stop_gradient)
        

def convert_to_meta_tensor(inputs):
    def func(x):
        if isinstance(x, ProxyTensor): 
            return x.meta
        return x
    return paddle.utils.map_structure(func, inputs)
    

def convert_to_symbol(inputs):
    def func(x):
        if isinstance(x, ProxyTensor): 
            return Symbol(x.name)
        return x
    pack_inputs = inputs
    if not paddle.utils.is_sequence(inputs):
        pack_inputs = [inputs]
    ret = paddle.utils.map_structure(func, pack_inputs)
    if not paddle.utils.is_sequence(inputs):
        ret = ret[0]
    return ret

def convert_arguments(inputs):
    #TODO (xionkgun): consider the following case: 
    # >>> x = [tensor1, tensor2, tensor3]
    # >>> paddle.stack(x)
    # we should convert tensor to proxy tensor here.
    def func(x):
        if isinstance(x, paddle.Tensor):
            return ProxyTensorContext().from_tensor(x)
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
        if func in [ paddle.add, paddle.subtract, paddle.nn.functional.relu ]:
            meta = infer_meta(func, *args)
            result = ProxyTensor(SymbolicTraceContext().new_varname(), meta)
            SymbolicTraceContext().call_API(func, inputs=convert_to_symbol(args), outputs=convert_to_symbol(result)) # symbolic only contain symbols.
            return result
        retval = func(*args)
        paddle.fluid.core.set_eval_frame(old_cb)
        return retval
    return wrapper
