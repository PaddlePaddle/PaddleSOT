import paddle
import inspect
from symbolic_trace import symbolic_trace
from symbolic_trace.utils import no_eval_frame, is_proxy_tensor
import unittest
import numpy as np

class TestCaseBase(unittest.TestCase):
    def assert_results(self, func, *inputs):
        sym_output = symbolic_trace(func)(*inputs)
        paddle_output = func(*inputs)
        np.testing.assert_allclose(
            sym_output, 
            paddle_output)

@no_eval_frame
def check_live_vars(has_value_vars, no_value_vars):
    current_frame = inspect.currentframe()
    assert current_frame is not None

    no_eval_frame_func_frame = current_frame.f_back
    assert no_eval_frame_func_frame is not None
    assert no_eval_frame_func_frame.f_code.co_name == "no_eval_frame_func"

    test_case_func_frame = no_eval_frame_func_frame.f_back
    assert test_case_func_frame is not None

    for has_value_var in has_value_vars:
        assert has_value_var in test_case_func_frame.f_locals
        proxy_tensor = test_case_func_frame.f_locals[has_value_var]
        if is_proxy_tensor(proxy_tensor):
            assert proxy_tensor.value() is not None, f"{has_value_var} ({proxy_tensor.name}) has no value"

    for no_value_var in no_value_vars:
        assert no_value_var in test_case_func_frame.f_locals
        proxy_tensor = test_case_func_frame.f_locals[no_value_var]
        if is_proxy_tensor(proxy_tensor):
            assert proxy_tensor.value() is None, f"{no_value_var} ({proxy_tensor.name}) has value {proxy_tensor.value()}"
