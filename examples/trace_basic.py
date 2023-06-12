import numpy as np

import paddle
from sot.translate import symbolic_translate


def foo(x: paddle.Tensor, y: paddle.Tensor):
    z = x + y
    return z + 1


def main():
    x = paddle.rand([2, 3])
    y = paddle.rand([2, 3])
    dygraph_out = foo(x, y)
    symbolic_translate_out = symbolic_translate(foo)(x, y)

    print("dygraph_out:", dygraph_out)
    print("symbolic_translate_out:", symbolic_translate_out)
    np.testing.assert_allclose(
        dygraph_out.numpy(), symbolic_translate_out.numpy()
    )


if __name__ == '__main__':
    main()
