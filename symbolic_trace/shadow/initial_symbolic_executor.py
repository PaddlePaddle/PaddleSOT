from .symbolic_executor import SymbolicExecutor
from .symbolic_frame_mgr import SymbolicFrameMgr
from ..utils import no_eval_frame
import types

class InitialSymbolicExecutor(SymbolicExecutor):
    @no_eval_frame
    def __init__(self, code_obj: types.CodeType):
        frame = SymbolicFrameMgr.current_frame(code_obj)
        super().__init__(frame)

    def pre_RETURN_VALUE(self, instruction):
        assert len(self.frame.stack) == 1, "Stack must have one element."
        ret_val = self.pop()
        new_code, guard_fn = self.frame.function_graph.start_compile(ret_val)
        from .symbolic_translator_cache import SymbolicTranslatorCache 
        SymbolicTranslatorCache().update_executed_code_obj(self.frame.f_code, new_code)
