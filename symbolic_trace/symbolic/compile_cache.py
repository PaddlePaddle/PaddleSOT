import paddle

from ..utils import Cache, Singleton
from .interpreter import compile_sir


@Singleton
class CompileSIRCache(Cache):
    def __init__(self):
        super().__init__(weak=False)

    def key_fn(self, context, sir_name):
        sir = context.get_sir(sir_name)
        # NOTE(dev): Is str(sir) a heavy opearation ?
        hash_key = hash(str(sir))
        return hash_key

    def value_fn(self, context, sir_name):
        return paddle.jit.to_static(
            compile_sir(context, sir_name), enable_fallback=False
        )
