import dataclasses
import paddle
from paddle.fluid.framework import Program

from .utils import NameGenerator, Singleton, no_eval_frame, meta_str


class MetaInfo: 
    def __init__(self, shape, dtype, stop_gradient):
        self.shape = shape
        self.dtype = dtype
        self.stop_gradient = stop_gradient

    @staticmethod
    def from_tensor(tensor):
        return MetaInfo(tensor.shape, tensor.dtype, tensor.stop_gradient)
    
    def __repr__(self):
        return meta_str(self.shape, self.dtype, self.stop_gradient)

    def __eq__(self, meta):
        shape_eq = (self.shape == meta.shape)
        dtype_eq = (self.dtype == meta.dtype)
        stop_gradient_eq = (self.stop_gradient == meta.stop_gradient)
        return shape_eq and dtype_eq and stop_gradient_eq


@Singleton
class VariableCreator:
    def __init__(self):
        self.var_cache = {}
        self.main_program = Program()
        self.startup_program = Program()
    
    def gen_name(self, meta):
        name = f"{meta.dtype}_{meta.stop_gradient}"
        for l in meta.shape:
            name += f"_{l}"
        return name

    def new_var(self, meta):
        var = self.main_program.global_block().create_var(
            shape = meta.shape,
            dtype = meta.dtype,
            stop_gradient = meta.stop_gradient,
        )
        return var

    def get_variable(self, meta):
        var_feature_name = self.gen_name(meta)

        if var_feature_name not in self.var_cache:
            self.var_cache[var_feature_name] = self.new_var(meta)
        return self.var_cache[var_feature_name]

    def infer_meta(self, func, *args, **kwargs):
        paddle.enable_static()
        args, kwargs = convert_to_variable(*args, **kwargs)

        with paddle.static.program_guard(self.main_program, self.startup_program):
            if isinstance(func, str):
                out = getattr(args[0], func)(*args[1:], **kwargs)
            else:
                out = func(*args, **kwargs)

        out = MetaInfo(
            list(out.shape),
            out.dtype,
            out.stop_gradient,
        )

        paddle.disable_static()
        return out


def convert_to_variable(*args, **kwargs):
    def func(x):
        if isinstance(x, MetaInfo):
            return VariableCreator().get_variable(x)
        return x
    return (paddle.utils.map_structure(func, args), 
            paddle.utils.map_structure(func, kwargs))


@no_eval_frame
def infer_meta(func, *args, **kwargs):
    return VariableCreator().infer_meta(func, *args, **kwargs)

