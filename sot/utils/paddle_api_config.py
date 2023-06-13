import inspect

import paddle


def get_tensor_methods():
    return [
        member_name
        for member_name, member in inspect.getmembers(paddle.static.Variable)
        if inspect.isfunction(member)
    ]


def get_paddle_api():
    modules = [
        paddle,
        paddle.nn.functional,
        paddle.linalg,
        paddle.signal,
        paddle.fft,
    ]
    non_operator_related_apis = [
        paddle.in_dynamic_mode,
        paddle.save,
        paddle.load,
        paddle.get_cuda_rng_state,
        paddle.set_rng_state,
        paddle.set_cuda_rng_state,
        paddle.get_rng_state,
        paddle.set_default_dtype,
        paddle.check_shape,
        paddle.summary,
        paddle.finfo,
        paddle.iinfo,
        paddle.enable_static,
        paddle.disable_static,
        paddle.is_grad_enabled,
    ]
    paddle_api_list = []
    for module in modules:
        for fn_name in getattr(module, "__all__", []):
            fn = getattr(module, fn_name)
            if inspect.isfunction(fn):
                paddle_api_list.append(fn)
    return list(set(paddle_api_list) - set(non_operator_related_apis))


paddle_tensor_methods = get_tensor_methods()
paddle_api_list = get_paddle_api()

# TODO(Aurelius84): It seems that we use it to judge 'in_paddle_module()'.
# Bug what does 'is_paddle_module' really means? Is all paddle.xx sub module
# considered as paddle module？
paddle_api_module_prefix = {
    "paddle.nn.functional",
    "paddle.nn.layer.activation",
}

break_graph_set = {
    print,
    paddle.to_tensor,  # TODO: paddle.to_tensor is not static/dygraph the same.
    paddle.grad,  # TODO(xiongkun): support paddle.grad.
    # paddle.utils.map_structure,
}


break_graph_tensor_method = {
    'register_hook',
    'numpy',
    'clear_gradient',
    # TODO: Browse all possible functions and make prior judgments.
}


def is_break_graph_tensor_methods(method_name):
    return method_name in break_graph_tensor_method


def add_break_graph_apis(apis: list):
    break_graph_set.update(apis)
