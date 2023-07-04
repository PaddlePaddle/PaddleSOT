from dataclasses import dataclass

from ..utils import Singleton, log

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


def add_breakpoint(file, line):
    BreakpointManager().add(file, line)
