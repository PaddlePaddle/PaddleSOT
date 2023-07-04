from dataclasses import dataclass

from ..opcode_translator.instruction_utils import instrs_info
from ..utils import Singleton, log
from .executor.opcode_executor import OpcodeExecutorBase

# this file is a debug utils files for quick debug
# >>> sot.add_breakpoint(file, line)
# >>> sot.remove_breakpoint(file, line)


@dataclass
class Breakpoint:
    file: str
    line: int

    def __hash__(self):
        return hash((self.file, self.line))


@Singleton
class BreakpointManager:
    def __init__(self):
        self.breakpoints = set()
        self.executors = OpcodeExecutorBase.call_stack
        self.active = 0

    def add(self, file, line):
        log(1, f"add breakpoint at {file}:{line}")
        self.breakpoints.add(Breakpoint(file, line))

    def rm(self, *args, **kwargs):
        # interactive use, we use abbreviate
        self.breakpoints()

    def hit(self, file, line):
        _breakpoint = Breakpoint(file, line)
        if _breakpoint in self.breakpoints:
            return True
        return False

    def locate(self, exe):
        for i, _e in enumerate(self.executors):
            if _e is exe:
                self.activate = i
                return
        raise RuntimeError("Not found executor.")

    def up(self):
        if self.activate == 0:
            return
        self.activate -= 1

    def down(self):
        if self.activate >= len(self.executors):
            return
        self.activate += 1

    def where(self):
        """
        display all inline calls.
        """
        pass

    def dis(self, range=5):
        """
        display all instruction code and source code.
        """
        print("displaying debug info...")
        cur_exe = self.executors[self.activate]
        lines = instrs_info(cur_exe._instructions)
        lasti = cur_exe._lasti
        print("\n".join(lines[max(lasti - range, 0) : lasti + range + 1]))
        # cur_exe._code = dis.dis()


def add_breakpoint(file, line):
    BM.add(file, line)


BM = BreakpointManager()
