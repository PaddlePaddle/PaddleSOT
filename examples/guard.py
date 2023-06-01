import paddle
from symbolic_trace.trace import symbolic_trace


def foo(x: paddle.Tensor, y: paddle.Tensor, z: paddle.Tensor):
    a = x + y
    a *= x
    return a, z


def main():
    a = paddle.rand([1])
    b = paddle.rand([2, 3])
    c = paddle.rand([4])
    d = paddle.rand([5, 6])
    e = paddle.rand([])
    sym_foo = symbolic_trace(foo)
    dygraph_out = foo(a, b, c)
    symbolic_trace_out = sym_foo(a, b, c)

    print("dygraph_out:", dygraph_out)
    print("symbolic_trace_out:", symbolic_trace_out)

    # cache hit
    sym_foo(a, b, d)

    # cache miss
    sym_foo(e, b, c)


if __name__ == '__main__':
    main()
