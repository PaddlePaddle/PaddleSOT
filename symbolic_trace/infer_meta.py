import paddle
from paddle.fluid.framework import Program

from .utils import NameGenerator, Singleton, no_eval_frame


class MetaInfo: 
    def __init__(self, shape, dtype, stop_gradient):
        self.shape = shape
        self.dtype = dtype
        self.stop_gradient = stop_gradient

    @staticmethod
    def from_tensor(tensor):
        return MetaInfo(tensor.shape, tensor.dtype, tensor.stop_gradient)
    
    def __repr__(self):
        return f"shape: {self.shape},  dtype: {self.dtype},  stop_gradient: {self.stop_gradient}"


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

    def infer_meta(self, func, *args):
        paddle.enable_static()
        args = convert_to_variable(args)

        with paddle.static.program_guard(self.main_program, self.startup_program):
            if isinstance(func, str):
                out = getattr(args[0], func)(*args[1:])
            else:
                out = func(*args)

        out = MetaInfo(
            list(out.shape),
            out.dtype,
            out.stop_gradient,
        )

        paddle.disable_static()
        return out


def convert_to_variable(inputs):
    def func(x):
        if isinstance(x, MetaInfo):
            return VariableCreator().get_variable(x)
        return x
    return paddle.utils.map_structure(func, inputs)


@no_eval_frame
def infer_meta(func, *args):
    return VariableCreator().infer_meta(func, *args)