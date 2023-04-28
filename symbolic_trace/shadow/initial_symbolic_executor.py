from .symbolic_executor import SymbolicExecutor
from .symbolic_frame_mgr import SymbolicFrameMgr
from ..utils import no_eval_frame
import types

class InitialSymbolicExecutor(SymbolicExecutor):
    @no_eval_frame
    def __init__(self, code_obj: types.CodeType):
        frame = SymbolicFrameMgr.current_frame(code_obj)
        super().__init__(frame)

    @no_eval_frame
    def __del__(self):
        super().__del__()
        print("InitialSymbolicExecutor.__del__")
