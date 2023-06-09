import paddle
from symbolic_trace.trace import symbolic_trace


def error_foo_inner(x: paddle.Tensor):
    x += 1
    print(x)
    y = x[111]
    return y


def error_foo(x: paddle.Tensor):
    return error_foo_inner(x)


def main():
    x = paddle.to_tensor([[0]])
    try:
        symbolic_trace(error_foo)(x)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
