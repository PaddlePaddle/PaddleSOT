import numpy as np

import paddle
from symbolic_trace.trace import symbolic_trace


def foo(x: paddle.Tensor):
    x += 1
    y = x + 1
    z = y * 2
    return z[0], x.shape


def error_foo(x: paddle.Tensor):
    x += 1
    y = x.undefined_attr
    z = y * 2
    return z[0], x.shape


def main():
    x = paddle.to_tensor([[0]])
    dygraph_out = foo(x)
    symbolic_trace_out = symbolic_trace(foo)(x)

    print("dygraph_out:", dygraph_out)
    print("symbolic_trace_out:", symbolic_trace_out)
    np.testing.assert_allclose(
        dygraph_out[0].numpy(), symbolic_trace_out[0].numpy()
    )
    try:
        symbolic_trace(error_foo)(x)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
