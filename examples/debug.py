import numpy as np

import paddle
from symbolic_trace.trace import symbolic_trace


def foo(cond: paddle.Tensor, x: paddle.Tensor):
    x += 1
    y = x + 1
    z = y * 2
    return z * 2


def main():
    cond = paddle.to_tensor(True)
    x = paddle.to_tensor(0)
    dygraph_out = foo(cond, x)
    symbolic_trace_out = symbolic_trace(foo)(cond, x)

    print("dygraph_out:", dygraph_out)
    print("symbolic_trace_out:", symbolic_trace_out)
    np.testing.assert_allclose(dygraph_out.numpy(), symbolic_trace_out.numpy())


if __name__ == '__main__':
    main()
