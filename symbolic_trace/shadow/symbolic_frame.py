from typing import List,Dict, Optional
from . import symbolic_frame_stack as symbolic_frame_stack
import types
import dis

class SymbolicFrame:
  f_locals: Dict[str, "VariableTracker"]
  function_graph: "FunctionGraph"
  f_code: types.CodeType
  stack: List["VariableTracker"]
  instructions: List[dis.Instruction]
  f_back: "SymbolicFrame"

  def __init__(self, f_locals, function_graph, code_obj, instructions):
    self.f_locals = f_locals
    self.function_graph = function_graph
    self.f_code = code_obj
    self.instructions = instructions
    self.stack = []
    self.f_back = symbolic_frame_stack.top()
    symbolic_frame_stack.push(self)

  def __del__(self):
    symbolic_frame_stack.pop(self.f_back)
