import paddle
import inspect
from symbolic_trace import symbolic_trace
from symbolic_trace.utils import no_eval_frame, is_proxy_tensor
from symbolic_trace.proxy_tensor import ProxyTensorContext
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
def check_live_vars(live_vars, dead_vars):
    current_frame = inspect.currentframe()
    assert current_frame is not None

    no_eval_frame_func_frame = current_frame.f_back
    assert no_eval_frame_func_frame is not None
    assert no_eval_frame_func_frame.f_code.co_name == "no_eval_frame_func"

    test_case_func_frame = no_eval_frame_func_frame.f_back
    assert test_case_func_frame is not None

    runtime_live_proxy_tensors = set(ProxyTensorContext().runtime_proxy_tensor_to_name.keys())
    runtime_live_eager_tensors = set(ProxyTensorContext().tensor_to_proxy_tensor.keys())

    for live_var in live_vars:
        assert live_var in test_case_func_frame.f_locals
        local_var = test_case_func_frame.f_locals[live_var]
        if is_proxy_tensor(local_var):
            proxy_tensor_id = id(local_var)
            assert proxy_tensor_id in runtime_live_proxy_tensors, f"{live_var} ({local_var.name}) is not live"
        elif isinstance(local_var, paddle.Tensor):
            eager_tensor_id = id(local_var)
            assert eager_tensor_id in runtime_live_eager_tensors, f"{live_var} ({local_var.name}) is not live"

    for dead_var in dead_vars:
        assert dead_var in test_case_func_frame.f_locals
        local_var = test_case_func_frame.f_locals[dead_var]
        print(dead_var, local_var)
        if is_proxy_tensor(local_var):
            proxy_tensor_id = id(local_var)
            assert proxy_tensor_id not in runtime_live_proxy_tensors, f"{dead_var} ({local_var.name}) is live"
        elif isinstance(local_var, paddle.Tensor):
            eager_tensor_id = id(local_var)
            assert eager_tensor_id not in runtime_live_eager_tensors, f"{dead_var} ({local_var.name}) is live"
