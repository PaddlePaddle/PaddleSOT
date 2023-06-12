import unittest

import paddle
from sot import symbolic_translate
from sot.proxy_tensor import ProxyTensor


def _ret_func():
    def inner():
        print("inner called")
        return paddle.to_tensor([1, 2, 3])

    return inner


def return_callable():
    retval = _ret_func()()
    assert isinstance(retval, ProxyTensor)


def _ret_tuple():
    print("i am called")
    return 1, 2, paddle.to_tensor([1, 2, 3])


def return_tuple():
    a, b, c = _ret_tuple()
    print(a, b, c)
    assert isinstance(c, ProxyTensor)


def val_in_container():
    mylist = [0, [_ret_tuple, 0], 1, 2, 3]
    a, b, c = mylist[1][0]()
    assert isinstance(c, ProxyTensor)


class TestCaseName(unittest.TestCase):
    def test_return_callable(self):
        symbolic_translate(return_callable)()

    def test_return_tuple(self):
        symbolic_translate(return_tuple)()

    def test_val_in_container(self):
        symbolic_translate(val_in_container)()


if __name__ == "__main__":
    unittest.main()
