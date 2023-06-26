import unittest


def normal_side_effect_1(tensor_x, list_a):
    tensor_x = tensor_x + 1
    list_a.append(12)
    return tensor_x, list_a


def normal_side_effect_2(tensor_x, list_a):
    tensor_x = tensor_x + 1
    list_a.append(tensor_x)
    return tensor_x, list_a


def normal_side_effect_3(list_a):
    """list index tracker."""
    del list_a[0]
    return list_a[0]


def normal_side_effect_5(list_a):
    """nested side effect"""
    inner_list = []
    inner_list.append(list_a)
    inner_list[-1].append(12)
    return 12
    # check list_a


a = 12


def normal_size_effect_6(tensor_x):
    """global"""
    global a
    a = 1
    return tensor_x + a


class CustomObject:
    def __init__(self):
        self.x = 0


def normal_size_effect_7(cus_obj, t):
    """object side effect."""
    t = t + 1
    cus_obj.x = t
    return t, cus_obj


# class TestNumpyAdd(TestCaseBase):
# @strict_mode_guard(0)
# def test_numpy_add(self):
# x = paddle.to_tensor([2])
# y = paddle.to_tensor([3])
# self.assert_results(numpy_add, x, y)


if __name__ == "__main__":
    unittest.main()
