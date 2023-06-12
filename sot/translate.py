import paddle

from .opcode_translator import eval_frame_callback


def symbolic_translate(func):
    def impl(*args, **kwargs):
        paddle.fluid.core.set_eval_frame(eval_frame_callback)
        try:
            outs = func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            paddle.fluid.core.set_eval_frame(None)
        return outs

    return impl
