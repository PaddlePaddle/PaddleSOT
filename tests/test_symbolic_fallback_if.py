import paddle
from symbolic_trace import symbolic_trace

def case1(cond, x):
    cond = 2 + cond
    if cond:
        print('yes')
        x = x + 4
    else:
        print("no")
        x = x - 4
    ret = paddle.nn.functional.relu(x)
    return ret


x = paddle.to_tensor([10.0])
print(symbolic_trace(case1)(paddle.to_tensor([-2]), x))
print(symbolic_trace(case1)(paddle.to_tensor([4.0]), x))
