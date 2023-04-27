import dis

import paddle
from symbolic_trace import symbolic_trace


def func(x, y):
    ret = 2 * x
    ret = paddle.nn.functional.relu(ret)
    ret = ret + y
    return ret


def simple(x):
    ret = 2 * x
    return ret


x = paddle.to_tensor([1.0])
y = paddle.to_tensor([2.0])


print(symbolic_trace(simple)(x))
print(symbolic_trace(simple)(y))
