import types
import dis
from typing import Dict, List
from .symbolic_translator import SymbolicTranslator

class SymbolicTranslatorCache:
    # TODO(tianchao): refactor to Dict[types.CodeType, GuardedFunctions]
    code_obj2translated_code_cache: Dict[types.CodeType, types.CodeType] = {}
    # TODO(tianchao): refactor to Dict[types.CodeType, GuardedFunctions]
    code_obj2executed_code_cache: Dict[types.CodeType, types.CodeType] = {}
    def __init__(self):
        pass

    def __call__(self, frame):
        code_obj = (
            self._find_executed_code_obj(frame)
            if self._has_executed_code_obj(frame)
            else self.find_or_translate(frame)
        )
        print('-'*100)
        dis.dis(code_obj)
        print('-'*100)
        return code_obj

    def find_or_translate(self, frame):
        origin_code_obj = frame.f_code
        if origin_code_obj not in type(self).code_obj2translated_code_cache:
            code_obj = SymbolicTranslator(frame)()
            type(self).code_obj2translated_code_cache[origin_code_obj] = code_obj
        return type(self).code_obj2translated_code_cache[origin_code_obj]

    def _find_executed_code_obj(self, frame):
        code_obj = type(self).code_obj2executed_code_cache[frame.f_code]
        return code_obj

    def _has_executed_code_obj(self, frame):
        return frame.f_code in type(self).code_obj2executed_code_cache

    def update_executed_code_obj(self, code_obj, new_code_obj):
        dis.dis(new_code_obj)
        type(self).code_obj2executed_code_cache[code_obj] = new_code_obj
