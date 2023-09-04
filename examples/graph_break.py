import numpy as np

import paddle
from sot.translate import symbolic_translate


def foo(cond: paddle.Tensor, x: paddle.Tensor):
    x += 1
    if cond:
        x += 1
    else:
        x -= 1
    return x


def main():
    cond = paddle.to_tensor(True)
    x = paddle.to_tensor(0)
    dygraph_out = foo(cond, x)
    symbolic_translate_out = symbolic_translate(foo)(cond, x)

    print("dygraph_out:", dygraph_out)
    print("symbolic_translate_out:", symbolic_translate_out)
    np.testing.assert_allclose(
        dygraph_out.numpy(), symbolic_translate_out.numpy()
    )


if __name__ == '__main__':
    main()
