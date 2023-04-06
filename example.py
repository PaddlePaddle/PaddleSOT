import paddle
from symbolic_trace import symbolic_trace

def origin_call():
    print("i am called")
    return 1,2,3

def ret_func():
    def inner():
        print("inner called")
    return inner


def caller():
    print("caller is calling")
    a,b,c = origin_call()
    print(a,b,c)
    ret_func()()
    mylist = [0, [origin_call, 0], 1,2,3]
    mylist[1][0]()
    tensor = paddle.to_tensor([1,2,3])
    print(type(tensor))


symbolic_trace(caller)()
