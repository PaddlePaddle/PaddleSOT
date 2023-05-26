import inspect
import json
import os
import sys
import warnings

import paddle


def get_tensor_methods():
    return [
        member_name
        for member_name, member in inspect.getmembers(paddle.static.Variable)
        if inspect.isfunction(member)
    ]


paddle_api_file_path = os.path.join(
    os.path.dirname(__file__), "paddle_api_info", "paddle_api.json"
)
with open(paddle_api_file_path, "r") as file:
    paddle_api = json.load(file)


paddle_api_list = set()
for module_name in paddle_api.keys():
    # it should already be imported
    if module_name in sys.modules.keys():
        module = sys.modules[module_name]
        apis = paddle_api[module_name]
        for api in apis:
            if api in module.__dict__.keys():
                obj = module.__dict__[api]
                paddle_api_list.add(obj)
    else:
        warnings.warn(f"{module_name} not imported.")

paddle_tensor_methods = get_tensor_methods()

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
