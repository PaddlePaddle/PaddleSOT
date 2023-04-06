import paddle
from symbolic_trace import symbolic_trace

def case1(x):
    kk = 'sdfsdf'
    y = 2 + x
    ret = paddle.nn.functional.relu(y)
    print(bool(y))
    print("yes")
    print("no")
    #for i in range(10):
    ret = ret + 2 + x
    return ret

x = paddle.to_tensor([1.0])
print(symbolic_trace(case1)(x))
