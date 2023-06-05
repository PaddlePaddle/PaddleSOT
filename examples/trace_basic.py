import numpy as np

import paddle
from symbolic_trace.trace import symbolic_trace


def foo(x: paddle.Tensor, y: paddle.Tensor):
    z = x + y
    return z + 1


def main():
    x = paddle.rand([2, 3])
    y = paddle.rand([2, 3])
    dygraph_out = foo(x, y)
    symbolic_trace_out = symbolic_trace(foo)(x, y)

    print("dygraph_out:", dygraph_out)
    print("symbolic_trace_out:", symbolic_trace_out)
    np.testing.assert_allclose(dygraph_out.numpy(), symbolic_trace_out.numpy())


if __name__ == '__main__':
    main()
