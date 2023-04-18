import paddle
import os, sys
import json
import warnings

paddle_api_file_path = os.path.join(os.path.dirname(__file__), "paddle_api_info", "paddle_api.json")
with open(paddle_api_file_path, "r") as file:
    paddle_api = json.load(file)

paddle_tensor_method_file_path = os.path.join(os.path.dirname(__file__), "paddle_api_info", "paddle_tensor_method.json")
with open(paddle_tensor_method_file_path, "r") as file:
    paddle_tensor_method = json.load(file)

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

paddle_api_module_prefix = set([
    'paddle.nn.functional', 
    'paddle.nn.layer.activation', 
])

fallback_list = set([
    print,
    #paddle.utils.map_structure,
])
