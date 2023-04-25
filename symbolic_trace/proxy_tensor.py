import paddle

from .infer_meta import InferMetaCache, MetaInfo
from .symbolic.statement_ir import Symbol

# from .symbolic.symbolic_context import SymbolicTraceContext
from .utils import (
    NameGenerator,
    Singleton,
    count_if,
    is_fallback_api,
    is_paddle_api,
    log,
    no_eval_frame,
)


# global variables
@Singleton
class ProxyTensorContext:
    def __init__(self):
        self.reset()

    def reset(self):
        self.runtime_name_to_proxy_tensor = {}
        self.runtime_proxy_tensor_to_name = {}
        self.tensor_to_proxy_tensor = {}
        self.var_name_generator = NameGenerator("var_")

    def new_varname(self):
        return self.var_name_generator.next()

    def from_tensor(self, tensor):
        # TODO: don't have the same name.
        if self.tensor_to_proxy_tensor.get(id(tensor), None) is not None:
            return self.tensor_to_proxy_tensor[id(tensor)]

        # TODO(id may have collision)
        name = self.new_varname()
        proxy_tensor = ProxyTensor(name, MetaInfo.from_tensor(tensor))
        self.tensor_to_proxy_tensor[id(tensor)] = proxy_tensor
        proxy_tensor.set_value(tensor)
        return proxy_tensor

    def bind_name_to_proxy_tensor(self, name, proxy_tensor):
        self.runtime_name_to_proxy_tensor[name] = proxy_tensor
        self.runtime_proxy_tensor_to_name[id(proxy_tensor)] = name

    def clear_proxy_tensor_by_name(self, name):
        log(3, f"[GC] trying to GC {name}\n")
        proxy_tensor = self.runtime_name_to_proxy_tensor[name]
        proxy_tensor_id = id(proxy_tensor)
        has_value = proxy_tensor.value() is not None
        eager_tensor_id = id(proxy_tensor.value())

        del self.runtime_name_to_proxy_tensor[name]
        del self.runtime_proxy_tensor_to_name[proxy_tensor_id]
        if has_value and eager_tensor_id in self.tensor_to_proxy_tensor:
            del self.tensor_to_proxy_tensor[eager_tensor_id]
        log(3, f"[GC] {name} GCed\n")

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

    @property
    def ndim(self):
        return len(self.meta)

    def set_value(self, value):
        """
        value is a eager tensor.
        when a proxytensor have value, it means it can be evaluated outer to_static.
        """
        self.value_ = value

    def clear_value(self):
        self.value_ = None

    def value(self):
        return self.value_

    # @staticmethod
    # def call_method(method_name, *args):
    #     args = convert_arguments(args)
    #     metas = convert_to_meta(args)
    #     meta = infer_meta(method_name, *metas)
    #     result = ProxyTensor(SymbolicTraceContext().new_varname(), meta)
    #     SymbolicTraceContext().call_METHOD(
    #         method_name,
    #         inputs=(convert_to_symbol(args), {}),
    #         outputs=convert_to_symbol(result),
    #     )  # symbolic only contain symbols.
    #     return result


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
    # TODO (xionkgun): consider the following case:
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
            if (
                count_if(
                    [args, kwargs], pred=lambda x: isinstance(x, ProxyTensor)
                )
                > 0
            ):
                args, kwargs = SymbolicTraceContext().start_compile(
                    ProxyTensorContext(), output=[args, kwargs]
                )
            # call function on eager tensor.
            return func(*args, **kwargs)

        elif is_paddle_api(func):
            # not fallback api, start symbolic trace.
            # TODO(xiokgun): multi-output support.
            # TODO(xiokgun): may have python buildin object inside metas.
            # TODO(xiokgun): 4 kinds of python arguments. support it !!
            log(3, f"call paddle.api : {func.__name__}", "\n")
            metas = convert_to_meta(args)
            kwmetas = convert_to_meta(kwargs)
            meta = InferMetaCache()(func, *metas, **kwmetas)
            result = ProxyTensor(ProxyTensorContext().new_varname(), meta)
            inputs_symbols = (
                convert_to_symbol(args),
                convert_to_symbol(kwargs),
            )
            log(3, f"         inputs : {inputs_symbols}", "\n")
            SymbolicTraceContext().call_API(
                func, inputs=inputs_symbols, outputs=convert_to_symbol(result)
            )  # symbolic only contain symbols.
            return result

        else:
            pass

    return wrapper
