import paddle
paddle_api_list = set([
    paddle.add,
    paddle.nn.functional.relu,
    paddle.to_tensor,
    paddle.concat, 
    paddle.split,
    paddle.subtract,
    paddle.nn.functional.common.linear,
])

fallback_list = set([
    print,
    #paddle.utils.map_structure,
])
