import paddle

from .opcode_translator import eval_frame_callback


def symbolic_translate(func, build_strategy=None):
    def callback(frame):
        return eval_frame_callback(frame, build_strategy)

    def impl(*args, **kwargs):
        paddle.fluid.core.set_eval_frame(callback)
        try:
            outs = func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            paddle.fluid.core.set_eval_frame(None)
        return outs

    return impl
