from .utils import no_eval_frame


# The MoneyPatch module adds methods to a class.
def proxy_tensor_method_builder(method_name):
    @no_eval_frame
    def __impl__(self, other):
        return self.call_method(method_name, self, other)

    return __impl__


def do_monkey_patch(cls, patch_names, method_builder):
    for method_name in patch_names:
        setattr(cls, method_name, method_builder(method_name))


binary_operator_methods = [
    '__add__',
    '__sub__',
    '__rsub__',
    '__radd__',
    '__mul__',
    '__rmul__',
    '__gt__',
    '__xor__',
    '__or__',
    '__and__',
    '__mod__',
    '__matmul__',
    '__pow__',
    '__floordiv__',
    '__truediv__',
]

unary_operator_methods = [
    '__invert__',
    '__neg__',
]
