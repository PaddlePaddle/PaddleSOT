import paddle
from symbolic_trace.trace import symbolic_trace


def error_foo(x: paddle.Tensor):
    x += 1
    print(x)
    y = x.undefined_attr
    z = y * 2
    return z[0], x.shape


def error_foo2(x: paddle.Tensor):
    x += 1
    z = y * 2  # noqa: F821
    return z


def main():
    x = paddle.to_tensor([[0]])
    try:
        symbolic_trace(error_foo)(x)
    except Exception as e:
        print(e)
    try:
        symbolic_trace(error_foo2)(x)
    except Exception as e:
        print(e)
        raise e


if __name__ == '__main__':
    main()
