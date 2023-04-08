import paddle

paddle_api_list = set([
    paddle.add,
    paddle.nn.functional.relu,
    paddle.to_tensor,
    paddle.concat, 
    paddle.split,
    paddle.subtract,
    paddle.flatten,
])

paddle_api_module_prefix = set([
    'paddle.nn.functional', 
    'paddle.nn.layer.activation', 
])

fallback_list = set([
    print,
    #paddle.utils.map_structure,
])
