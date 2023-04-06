import paddle
from .symbolic_trace import SymbolicTraceContext
from .statement_ir import Symbol
from .utils import Singleton, no_eval_frame
from .infer_meta import infer_meta, MetaInfo

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

    @no_eval_frame
    def __bool__(self):
        # TODO: (too ugly, need to be refactored)
        SymbolicTraceContext().start_compile(ProxyTensorContext().get_runtime())
        assert self.value() is not None
        return bool(self.value())

    @staticmethod
    def call_method(method_name, *args):
        args = convert_arguments(args)
        if method_name in [ "__add__", "__radd__", "__sub__", "__rsub__" ]:
            metas = convert_to_meta(args)
            meta = infer_meta(method_name, *metas)
            result = ProxyTensor(SymbolicTraceContext().new_varname(), meta)
            SymbolicTraceContext().call_METHOD(method_name, inputs=convert_to_symbol(args), outputs=convert_to_symbol(result)) # symbolic only contain symbols.
            return result

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
    pack_inputs = inputs
    if not paddle.utils.is_sequence(inputs):
        pack_inputs = [inputs]
    ret = paddle.utils.map_structure(func, pack_inputs)
    if not paddle.utils.is_sequence(inputs):
        ret = ret[0]
    return ret


@no_eval_frame
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


@no_eval_frame
def paddle_api_wrapper(func):
    @no_eval_frame
    def wrapper(*args): 
        args = convert_arguments(args)
        # TODO(xiokgun): multi-output support.
        # TODO(xiokgun): may have python buildin object inside metas.
        # TODO(xiokgun): 4 kinds of python arguments. support it !!
        if func in [ paddle.add, paddle.subtract, paddle.nn.functional.relu ]:
            metas = convert_to_meta(args)
            meta = infer_meta(func, *metas)
            result = ProxyTensor(SymbolicTraceContext().new_varname(), meta)
            SymbolicTraceContext().call_API(func, inputs=convert_to_symbol(args), outputs=convert_to_symbol(result)) # symbolic only contain symbols.
            return result
        retval = func(*args)
        return retval
    return wrapper
