import inspect
import types
import unittest

from test_case_base import TestCaseBase, strict_mode_guard

import paddle
from sot.psdb import check_no_breakgraph


def foo(x: int, y: paddle.Tensor):
    z = 3

    def local(a, b=5):
        return a + x + z + b + y

    return local(4) + z


def foo2(y: paddle.Tensor, x=1):
    """
    Test strip default value
    """
    z = 3

    def local(a, b=5):
        return a + x + z + b + y

    return local(4)


def foo3(y: paddle.Tensor, x=1):
    """
    Test Closure Band Default
    """
    z = 3

    def local(a, b=5):
        nonlocal z
        z = 4
        return a + x + z + b + y

    return local(4)


global_z = 3


def test_global(y: paddle.Tensor):
    """
    Test Global variable
    """

    def local(a, b=5):
        global global_z
        global_z += 1
        return a + global_z + b + y

    return local(1)


def multi(c):
    return c + 2


def wrapper_function(func):
    a = 2

    def inner():
        return func(a)

    return inner


wrapped_multi = wrapper_function(multi)


def foo5(y: paddle.Tensor):
    """
    Test incoming closures
    """
    a = wrapped_multi()
    return a


def outwrapper(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def foo6(y: paddle.Tensor):
    """
    Test Decorator
    """

    @outwrapper
    def load_1(a, b=5):
        return a + b

    return load_1(1)


@check_no_breakgraph
def closure_del():
    x = 0

    def load():
        nonlocal x
        del x

    return load


import numpy as np


def numpy_sum(m):
    """
    Test loop call

    Example: a->b->c->a
    """
    a = np.array([1, 2, 3])
    tmp = np.sum(a)
    return m + 1


def lambda_closure(x, m):
    """
    lambda closure.
    """

    def break_graph_closure():
        print("yes")
        return x + m

    return break_graph_closure()


# motivated by python builtin decorator
def kwargs_wrapper(func):
    sig = inspect.signature(func)

    def inner(*args, **kwargs):
        return func(*args, **kwargs)

    inner.__signature__ = sig
    return inner


@kwargs_wrapper
def func7(a, b):
    return a + b


def test_builtin_decorator():
    return func7(3, 5)


def create_closure():
    x = 1

    def closure():
        return x + 1

    return closure


def non_local_test(t: paddle.Tensor):
    a = 1

    def func1():
        nonlocal a
        t = a
        a = 2
        return t

    def func2():
        nonlocal a
        a = 1
        return a

    t += func1()  # add 2
    t += func2()  # add 1
    t += a  # add 1
    return t


# Side Effect.
def test_slice_in_for_loop(x, iter_num=3):
    x = paddle.to_tensor(x)
    a = []
    # Use `paddle.full` so that static analysis can analyze the type of iter_num is Tensor
    iter_num = paddle.full(
        shape=[1], fill_value=iter_num, dtype="int32"
    )  # TODO(liym27): Delete it if the type of parameter iter_num can be resolved

    for i in range(iter_num):
        a.append(x)

    for i in range(iter_num):
        a[i] = x
    out = a[2]
    return out


class TestClosure(TestCaseBase):
    def test_closure(self):
        self.assert_results(foo, 1, paddle.to_tensor(2))
        self.assert_results(foo2, paddle.to_tensor(2))
        self.assert_results(foo3, paddle.to_tensor(2))
        self.assert_results(foo5, paddle.to_tensor(2))

    def test_global(self):
        self.assert_results_with_global_check(
            test_global, ["global_z"], paddle.to_tensor(2)
        )

    def test_lambda(self):
        with strict_mode_guard(0):
            self.assert_results(
                lambda_closure, paddle.to_tensor(2), paddle.to_tensor(1)
            )

    def test_numpy(self):
        self.assert_results(numpy_sum, paddle.to_tensor(1))

    def test_del_deref(self):
        def is_empty_cell(cell: types.CellType):
            try:
                cell.cell_contents  # noqa: B018
                return False
            except ValueError as e:
                if "Cell is empty" in str(e):
                    return True
                return False

        closure_del_func = closure_del()
        # Why is it 0: Only one value x is stored in the closure_del method __Closure__ in
        self.assertFalse(is_empty_cell(closure_del_func.__closure__[0]))
        closure_del_func()
        self.assertTrue(is_empty_cell(closure_del_func.__closure__[0]))

    def test_decorator(self):
        self.assert_results(test_builtin_decorator)
        self.assert_results(foo6, paddle.to_tensor(2))

    def test_nolocal(self):
        tx = paddle.to_tensor([1.0])
        self.assert_results(non_local_test, tx)

    def test_side_effect(self):
        tx = paddle.to_tensor([1.0, 2.0, 3.0])
        # need side effect of list.
        self.assert_results_with_side_effects(test_slice_in_for_loop, tx)

    def test_create_closure(self):
        closure = create_closure()
        self.assert_results(closure)


if __name__ == "__main__":
    unittest.main()

# Instructions:
# LOAD_CLOSURE
# LOAD_DEREF
# LOAD_CLASSDEREF
# STORE_DEREF
# DELETE_DEREF
# STORE_GLOBAL
