import types
import dis
from typing import Tuple
from . import symbolic_frame_stack as symbolic_frame_stack
from .symbolic_frame import SymbolicFrame
from ..opcode_translator.executor.variables import VariableTrackerFactory
from ..opcode_translator.executor.function_graph import FunctionGraph

class SymbolicFrameMgr:
    @staticmethod
    def create_initial_frame(py_frame: types.FrameType):
        code_obj = py_frame.f_code
        arg_varnames = SymbolicFrameMgr._get_arg_varnames(code_obj)
        function_graph = FunctionGraph(py_frame.f_globals, py_frame.f_code)
        f_locals = SymbolicFrameMgr._make_f_locals_from_python_frame(
            py_frame, function_graph, arg_varnames
        )
        instructions = list(dis.get_instructions(code_obj))
        return SymbolicFrame(f_locals, function_graph, code_obj, instructions)

    @staticmethod
    def current_frame():
        frame = symbolic_frame_stack.top()
        assert frame is not None
        return frame
  
    @staticmethod
    def create_frame(py_frame: types.FrameType):
        code_obj = py_frame.f_code
        arg_varnames = SymbolicFrameMgr._get_arg_varnames(code_obj)
        f_locals = SymbolicFrameMgr._make_f_locals_from_symbolic_frame(arg_varnames)
        function_graph = symbolic_frame_stack.top().function_graph
        instructions = list(dis.get_instructions(code_obj))
        return SymbolicFrame(f_locals, function_graph, code_obj, instructions)

    @staticmethod
    def _make_f_locals_from_py_frame(frame: types.FrameType):
        arg_varnames = SymbolicFrameMgr._get_arg_varnames(code_obj)

    @staticmethod
    def _get_arg_varnames(code_obj: types.CodeType):
        kPosArgBit = 3
        assert code_obj.co_flags & (1 << kPosArgBit) == 0, "positional args are not supported yet."
        kKwArgBit = 4
        assert code_obj.co_flags & (1 << kKwArgBit) == 0, "keyword args are not supported yet."
        return code_obj.co_varnames[:code_obj.co_argcount]


    @staticmethod
    def _make_f_locals_from_python_frame(
            frame: types.FrameType,
            function_graph: FunctionGraph,
            arg_varnames: Tuple[str]
        ):
        return {
            arg_varname:VariableTrackerFactory.from_value(frame.f_locals[arg_varname], function_graph)
            for arg_varname in arg_varnames
        }

    @staticmethod
    def _make_f_locals_from_symbolic_frame(arg_varnames: Tuple[str]):
        assert len(symbolic_frame_stack.top().stack) >= len(arg_varnames)
        arg_vars = symbolic_frame_stack.top().stack[-len(arg_varnames):]
        return {arg_varnames[i]:arg_vars[i] for i in range(len(arg_varnames))}
