import inspect
from enum import Enum

from .utils import Singleton, log


class CodeState(Enum):
    UNKNOW = 1
    WITH_GRAPH = 2
    WITHOUT_GRAPH = 3


class CodeInfo:
    def __init__(self):
        self.state = CodeState.UNKNOW
        self.counter = 0

    def __repr__(self):
        return f"state: {self.state}, counter: {self.counter}"


@Singleton
class CodeStatus:
    def __init__(self):
        self.code_map = {}

    def clear(self):
        self.code_map.clear()

    def check_code(self, code):
        if code not in self.code_map:
            info = CodeInfo()
            self.code_map[code] = info
        else:
            info = self.code_map[code]

        if info.state == CodeState.WITHOUT_GRAPH:
            return True
        elif info.state == CodeState.UNKNOW:
            self.visit(code)
        return False

    def visit(self, code):
        info = self.code_map[code]
        info.counter += 1
        if info.state == CodeState.UNKNOW and info.counter > 10:
            log(3, f"[CodeStatus] Switch state to WITHOUT_GRAPH for {code}")
            info.state = CodeState.WITHOUT_GRAPH

    def trace_back_frames(self):
        frame = inspect.currentframe()
        while frame.f_back is not None:
            frame = frame.f_back
            code = frame.f_code
            if code in self.code_map:
                info = self.code_map[code]
                if info.state != CodeState.WITH_GRAPH:
                    log(
                        3, f"[CodeStatus] Switch state to WITH_GRAPH for {code}"
                    )
                    info.state = CodeState.WITH_GRAPH
