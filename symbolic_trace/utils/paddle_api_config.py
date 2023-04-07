import paddle
paddle_api_list = set([
    paddle.add,
    paddle.nn.functional.relu,
    paddle.to_tensor,
    paddle.concat, 
    paddle.split,
    paddle.subtract,
])

fallback_list = set([
])
