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
    paddle_api_list = []
    for module in modules:
        for fn_name in getattr(module, "__all__", []):
            fn = getattr(module, fn_name)
            if inspect.isfunction(fn):
                paddle_api_list.append(fn)
    return list(set(paddle_api_list))


paddle_tensor_methods = get_tensor_methods()
paddle_api_list = get_paddle_api()

# TODO(Aurelius84): It seems that we use it to judge 'in_paddle_module()'.
# Bug what does 'is_paddle_module' really means? Is all paddle.xx sub module
# considered as paddle module？
paddle_api_module_prefix = {
    "paddle.nn.functional",
    "paddle.nn.layer.activation",
}

fallback_list = {
    print,
    # paddle.utils.map_structure,
}
