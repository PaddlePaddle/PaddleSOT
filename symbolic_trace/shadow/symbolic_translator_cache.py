import types
import dis
from typing import Dict, List
from .symbolic_translator import SymbolicTranslator

class SymbolicTranslatorCache:
    # TODO(tianchao): refactor to Dict[types.CodeType, GuardedFunctions]
    code_obj2functions_cache: Dict[types.CodeType, types.CodeType] = {}
    def __init__(self):
        pass

    def __call__(self, frame):
        origin_code_obj = frame.f_code
        if origin_code_obj not in type(self).code_obj2functions_cache:
            code_obj = self.translate(frame)
            type(self).code_obj2functions_cache[origin_code_obj] = code_obj
            dis.dis(code_obj)
        return type(self).code_obj2functions_cache[origin_code_obj]

    def translate(self, frame):
        return SymbolicTranslator(frame)()
