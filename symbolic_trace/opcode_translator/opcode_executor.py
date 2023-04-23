from ..utils import log
from ..utils import InnerError, UnsupportError
import types
import dis

class OpcodeExecutor: 
    def __init__(self, frame: types.FrameType, code_options):
        self._code_options = code_options
        self._frame = frame
        self._stack = []
        self._code = frame.f_code
        self._locals = {}
        self._globals = {}
        self._lasti = 0  # idx of instruction list

        # Instructions is struture like the following: 
        # Instruction(opname='LOAD_CONST', 
        #             opcode=100, arg=1, argval=2, argrepr='2', offset=0, 
        #             starts_line=11, is_jump_target=False)
        self._instructions = list(dis.get_instructions(self._code))
        # offset -> instruction
        self.offset_map = {}

    def run(self):
        log(3, "start execute opcode: {self._code}")
        self._lasti = 0
        while True:
            if self._lasti >= len(self._instructions):
                raise InnerError("lasti out of range, InnerError.")
            cur_instr = self._instructions[self._lasti]
            self._lasti += 1
            is_stop = self.step(cur_instr)
            if is_stop: break

    def step(self, instr):
        if not hasattr(self, instr.opname):
            raise UnsupportError(f"opcode: {instr.opname} is not supported.")
        getattr(self, instr.opname)(instr) # run single step.

    def LOAD_ATTR(self, instr):
        pass
        
    def LOAD_FAST(self, instr):
        pass

    def BINARY_ADD(self, instr):
        pass

    def STORE_FAST(self, instr):
        pass

    def LOAD_GLOBAL(self, instr):
        pass
