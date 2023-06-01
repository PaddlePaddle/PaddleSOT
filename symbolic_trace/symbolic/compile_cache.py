import paddle

from ..utils import Cache, Singleton
from .interpreter import compile_sir


class FallbackWrapper:
    def __init__(self, compile_sir):
        self.compile_sir = compile_sir
        self.partial_program_layer = None

    def __call__(self, *args, **kwargs):
        frame_callback = paddle.fluid.core.set_eval_frame(None)
        if self.partial_program_layer is None:
            outputs = self.compile_sir(*args, **kwargs)
            self.partial_program_layer = self.compile_sir.get_concrete_program(
                *args, **kwargs
            )[1]
        else:
            # Speed up Resnet from 0.0068 --> 0.0057
            outputs = self.partial_program_layer(*args, **kwargs)
        paddle.fluid.core.set_eval_frame(frame_callback)
        return outputs


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
        return FallbackWrapper(
            paddle.jit.to_static(compile_sir(context, sir_name))
        )
