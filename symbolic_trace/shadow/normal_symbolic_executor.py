from .symbolic_executor import SymbolicExecutor
from .symbolic_frame_mgr import SymbolicFrameMgr
from ..utils import no_eval_frame
import types

class NormalSymbolicExecutor(SymbolicExecutor):
    @no_eval_frame
    def __init__(self, code_obj: types.CodeType):
        frame = SymbolicFrameMgr.create_frame(code_obj)
        super().__init__(frame)

    @no_eval_frame
    def __del__(self):
        super().__del__()
        print("NormalSymbolicExecutor.__del__")
