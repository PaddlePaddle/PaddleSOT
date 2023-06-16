# TODO(zrr1999): the file should be removed before merge


import paddle


def foo(cond: paddle.Tensor, x: paddle.Tensor):
    x += 1
    y = x.dim()
    print(y)
    return x + x.shape[0]


def main():
    cond = paddle.to_tensor(True)
    x = paddle.to_tensor([0])
    dygraph_out = foo(cond, x)
    print("dygraph_out:", dygraph_out)

    # symbolic_translate_out = symbolic_translate(foo)(cond, x)
    # print("symbolic_translate_out:", symbolic_translate_out)
    # np.testing.assert_allclose(
    #     dygraph_out.numpy(), symbolic_translate_out.numpy()
    # )


if __name__ == '__main__':
    main()
