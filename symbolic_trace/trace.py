import paddle

from .infer_meta import VariableCreator
from .opcode_translator import eval_frame_callback
from .opcode_translator.executor.opcode_executor import (
    InstructionTranslatorCache,
)
from .proxy_tensor import ProxyTensorContext
from .symbolic.compile_cache import CompileSIRCache
from .symbolic.statement_ir import SIRRuntimeCache, StatementIRFactory


def symbolic_trace(func):
    def impl(*args, **kwargs):
        ProxyTensorContext().reset()
        paddle.fluid.core.set_eval_frame(eval_frame_callback)
        try:
            outs = func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            paddle.fluid.core.set_eval_frame(None)
            ProxyTensorContext().reset()
            InstructionTranslatorCache().clear()
            VariableCreator().clear()
            CompileSIRCache().clear()
            StatementIRFactory().clear()
            SIRRuntimeCache().clear()
        return outs

    return impl
