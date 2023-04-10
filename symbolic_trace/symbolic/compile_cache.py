from ..utils import Cache
from .interpreter import compile_sir
from ..utils import Singleton
import paddle

@Singleton
class CompileSIRCache(Cache):
    def __init__(self):
        super().__init__(weak=False)

    def key_fn(self, context, sir_name):
        sir = context.get_sir(sir_name)
        hash_key = hash(str(sir))
        return hash_key

    def value_fn(self, context, sir_name):
        return paddle.jit.to_static(compile_sir(context, sir_name))
