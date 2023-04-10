import paddle
from .symbolic.symbolic_context import SymbolicTraceContext
from .symbolic.statement_ir import Symbol
from .utils import Singleton, no_eval_frame, is_paddle_api, is_fallback_api, log, count_if, map_if
from .opcode_translator import eval_frame_callback
from .infer_meta import infer_meta, MetaInfo
from .opcode_translator import ConvertGuard

def method_with_fallback(func):
    @no_eval_frame
    def fallback_inner(self, *args, **kwargs):
        SymbolicTraceContext().start_compile(
            ProxyTensorContext(), output=self
        )
        assert self.value() is not None
        ret = func(self, *args, **kwargs)
        return ret
    return fallback_inner

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
        self.tensor_to_proxy_tensor[id(tensor)] = proxy_tensor
        proxy_tensor.set_value(tensor)
        return proxy_tensor

    def bind_name_to_proxy_tensor(self, name, proxy_tensor):
        self.runtime_name_to_proxy_tensor[name] = proxy_tensor
        self.runtime_proxy_tensor_to_name[id(proxy_tensor)] = name 

    def get_runtime(self):
        return self.runtime_name_to_proxy_tensor


class ProxyTensor:
    def __init__(self, name, meta):
        self.name = name
        self.meta = meta
        self.value_ = None
        self._proxy_tensor_ = True
        ProxyTensorContext().bind_name_to_proxy_tensor(name, self)

    @property
    def shape(self):
        # TODO(xiongkun) consider dynamic shape.
        return self.meta.shape

    @no_eval_frame
    def set_value(self, value):
        """
        value is a eager tensor. 
        when a proxytensor have value, it means it can be evaluated outer to_static.
        """
        self.value_ = value

    @no_eval_frame
    def value(self):
        return self.value_
    
    @no_eval_frame
    def __add__(self, other):
        return self.call_method("__add__", self, other)

    @no_eval_frame
    def __gt__(self, other):
        return self.call_method("__gt__", self, other)

    @no_eval_frame
    def __lt__(self, other):
        return self.call_method("__lt__", self, other)

    @no_eval_frame
    def __eq__(self, other):
        return self.call_method("__lt__", self, other)

    @no_eval_frame
    def __mul__(self, other):
        return self.call_method("__mul__", self, other)

    @no_eval_frame
    def __radd__(self, other):
        return self.call_method("__radd__", self, other)

    @no_eval_frame
    def __sub__(self, other):
        return self.call_method("__sub__", self, other)

    @no_eval_frame
    def __rsub__(self, other):
        return self.call_method("__rsub__", self, other)

    @method_with_fallback
    def __bool__(self):
        return bool(self.value())

    @method_with_fallback
    def __int__(self):
        return int(self.value())

    @method_with_fallback
    def __iter__(self):
        # if we don't use a ProxyIterator, the eager tensor iter will put
        # in the stack. Calling in eval_frame_callback will cause errors.
        class ProxyIterator:
            def __init__(self, eager_iter):
                self.eager_tensor_iter = eager_iter

            @no_eval_frame # important
            def __next__(self):
                return next(self.eager_tensor_iter)
        return ProxyIterator(iter(self.value()))

    #TODO(xiongkun): cause error ????, why ?
    #@method_with_fallback
    #def __str__(self):
        #return self.value().__str__()

    #@method_with_fallback
    #def __repr__(self):
        #return self.value().__repr__()

    @method_with_fallback
    def numpy(self):
        return self.value().numpy()

    @staticmethod
    def call_method(method_name, *args):
        args = convert_arguments(args)
        metas = convert_to_meta(args)
        meta = infer_meta(method_name, *metas)
        result = ProxyTensor(SymbolicTraceContext().new_varname(), meta)
        SymbolicTraceContext().call_METHOD(
            method_name, 
            inputs=(convert_to_symbol(args), {}), 
            outputs=convert_to_symbol(result)) # symbolic only contain symbols.
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
def callable_wrapper(func):
    @no_eval_frame
    def wrapper(*args, **kwargs): 
        args, kwargs = convert_arguments(args), convert_arguments(kwargs) 
        if is_fallback_api(func):
            # fallback api, fallback first and call this api.
            if count_if([args, kwargs], pred=lambda x: isinstance(x, ProxyTensor)) > 0:
                SymbolicTraceContext().start_compile(
                    ProxyTensorContext(), output=[args, kwargs])
            # call function on eager tensor.
            args, kwargs = paddle.utils.map_structure(lambda x: x.value() if isinstance(x, ProxyTensor) else x, [args, kwargs])
            return func(*args, **kwargs)

        elif is_paddle_api(func):
            # not fallback api, start symbolic trace.
            # TODO(xiokgun): multi-output support.
            # TODO(xiokgun): may have python buildin object inside metas.
            # TODO(xiokgun): 4 kinds of python arguments. support it !!
            log(3, f"call paddle.api : {func.__name__}", "\n")
            metas = convert_to_meta(args)
            kwmetas = convert_to_meta(kwargs)
            meta = infer_meta(func, *metas, **kwmetas)
            result = ProxyTensor(SymbolicTraceContext().new_varname(), meta)
            inputs_symbols = (convert_to_symbol(args), convert_to_symbol(kwargs))
            log(3, f"         inputs : {inputs_symbols}", "\n")
            SymbolicTraceContext().call_API(
                func, 
                inputs=inputs_symbols,
                outputs=convert_to_symbol(result)) # symbolic only contain symbols.
            return result

        else: 
            pass
    return wrapper

@no_eval_frame
def cache_and_return(name, inputs):
    sir_name, _, full_outputs_with_tensor_meta = SymbolicTraceContext().statement_factory.cached_SIR[name]
    cur_sir = SymbolicTraceContext().statement_factory[sir_name]

    flat_inputs = paddle.utils.flatten(inputs)
    symbol_inputs = [Symbol(x.name) for x in flat_inputs if isinstance(x, ProxyTensor)]

    outputs = gen_new_proxy_tensor_output(cur_sir, full_outputs_with_tensor_meta)

    flat_outputs = paddle.utils.flatten(outputs)
    symbol_outputs = [Symbol(x.name) for x in flat_outputs if isinstance(x, ProxyTensor)]

    SymbolicTraceContext().call_SIR(cur_sir.name, symbol_inputs, symbol_outputs)
    return outputs

@no_eval_frame
# should generate a unique name for every funtion
def frame_enter(name, inputs):
    # need a better hash strategy
    key_list = []
    for inp in paddle.utils.flatten(inputs):
        if isinstance(inp, ProxyTensor):
            key_list.append(str(inp.meta))
        else:
            key_list.append(inp)

    cur_key = hash(tuple(key_list))

    if name in SymbolicTraceContext().statement_factory.cached_SIR.keys():
        _, input_hash, _ = SymbolicTraceContext().statement_factory.cached_SIR[name]
        if cur_key == input_hash:
            return True

    new_sir = SymbolicTraceContext().statement_factory.create()
    flat_inputs = paddle.utils.flatten(inputs)
    new_sir.inputs = [Symbol(x.name) for x in flat_inputs if isinstance(x, ProxyTensor)]
    setattr(new_sir, "func_name", name)
    setattr(new_sir, "input_hash", cur_key)
    SymbolicTraceContext().sir_stack.append(new_sir)
    return False

@no_eval_frame
def frame_leave(outputs):
    cur_sir = SymbolicTraceContext().sir_stack[-1]
    SymbolicTraceContext().sir_stack.pop()

    # gen symbol outputs for SIR
    flat_outputs = paddle.utils.flatten(outputs)
    cur_sir.outputs = [Symbol(x.name) for x in flat_outputs if isinstance(x, ProxyTensor)]

    # at the first time, the inputs and outputs need not change
    SymbolicTraceContext().call_SIR(cur_sir.name, cur_sir.inputs, cur_sir.outputs)

    # gen outputs with python value for SIR and cache SIR
    full_outputs_with_tensor_meta = map_if(outputs, pred=lambda x: isinstance(x, ProxyTensor), true_fn=lambda x: x.meta, false_fn=lambda x: x)
    SymbolicTraceContext().statement_factory.cached_SIR[cur_sir.func_name] = (cur_sir.name, cur_sir.input_hash, full_outputs_with_tensor_meta)
    print(SymbolicTraceContext().statement_factory.cached_SIR[cur_sir.func_name])
    # just return the outputs

def gen_new_proxy_tensor_output(sir, full_outputs_with_tensor_meta):
    # need to find inplace operate with sir, but it is not used now
    def create_new_proxy_tensor_from_meta(meta):
        result = ProxyTensor(SymbolicTraceContext().new_varname(), meta)
        return result
    return map_if(full_outputs_with_tensor_meta, pred=lambda x: isinstance(x, MetaInfo), true_fn=create_new_proxy_tensor_from_meta, false_fn=lambda x: x)