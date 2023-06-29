import unittest

from sot.opcode_translator.executor.mutable_data import (
    MutableData,
    MutableDictLikeData,
    MutableListLikeData,
)


class VariableBase:
    def __init__(self):
        ...


class ConstVariable(VariableBase):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"ConstVariable({self.value})"

    def __eq__(self, other):
        if not isinstance(other, ConstVariable):
            return False
        return self.value == other.value


class DictVariable(VariableBase):
    def __init__(self, data):
        self.data = data
        self.proxy = MutableDictLikeData(data, DictVariable.proxy_getter)

    @staticmethod
    def proxy_getter(data, key):
        if key not in data:
            return MutableData.Empty()
        return ConstVariable(data[key])

    def getitem(self, key):
        res = self.proxy.get(key)
        if isinstance(res, MutableData.Empty):
            raise KeyError(f"Key {key} not found")
        return res

    def setitem(self, key, value):
        self.proxy.set(key, value)

    def delitem(self, key):
        self.proxy.delete(key)


class ListVariable(VariableBase):
    def __init__(self, data):
        self.data = data
        self.proxy = MutableListLikeData(data, ListVariable.proxy_getter)

    @staticmethod
    def proxy_getter(data, key):
        if key < 0 or key >= len(data):
            return MutableData.Empty()
        return ConstVariable(data[key])

    def getitem(self, key):
        res = self.proxy.get(key)
        if isinstance(res, MutableData.Empty):
            raise IndexError(f"Index {key} out of range")
        return res

    def setitem(self, key, value):
        self.proxy.set(key, value)

    def delitem(self, key):
        self.proxy.delete(key)

    def insert(self, index, value):
        self.proxy.insert(index, value)

    def append(self, value):
        self.proxy.insert(self.proxy.length, value)

    def extend(self, value):
        for item in value:
            self.append(item)

    def pop(self, index=-1):
        res = self.getitem(index)
        self.delitem(index)
        return res

    def clear(self):
        for i in range(self.proxy.length):
            self.delitem(0)

    def remove(self, value):
        for i in range(self.proxy.length):
            if self.getitem(i) == value:
                self.delitem(i)
                return
        raise ValueError(f"Value {value} not found")

    def sort(self, key=None, reverse=False):
        if key is None:
            key = lambda x: x
        permutation = list(range(self.proxy.length))
        permutation.sort(
            key=lambda x: key(self.getitem(x).value), reverse=reverse
        )
        self.proxy.permutate(permutation)

    def reverse(self):
        permutation = list(range(self.proxy.length))
        permutation.reverse()
        self.proxy.permutate(permutation)


class TestMutableDictLikeVariable(unittest.TestCase):
    def test_getitem(self):
        data = {"a": 1, "b": 2}
        var = DictVariable(data)
        self.assertEqual(var.getitem("a"), ConstVariable(1))
        self.assertEqual(var.getitem("b"), ConstVariable(2))

    def test_setitem(self):
        data = {"a": 1, "b": 2}
        var = DictVariable(data)
        var.setitem("a", ConstVariable(3))
        self.assertEqual(var.getitem("a"), ConstVariable(3))
        var.setitem("c", ConstVariable(4))
        self.assertEqual(var.getitem("c"), ConstVariable(4))

    def test_delitem(self):
        data = {"a": 1, "b": 2}
        var = DictVariable(data)
        var.delitem("a")
        with self.assertRaises(KeyError):
            var.getitem("a")

    def test_keys(self):
        data = {"a": 1, "b": 2}
        var = DictVariable(data)
        self.assertEqual(list(var.proxy.get_all().keys()), ["a", "b"])


class TestMutableListLikeVariable(unittest.TestCase):
    def test_getitem(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        self.assertEqual(var.getitem(0), ConstVariable(1))
        self.assertEqual(var.getitem(1), ConstVariable(2))
        self.assertEqual(var.getitem(2), ConstVariable(3))

    def test_setitem(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.setitem(0, ConstVariable(4))
        self.assertEqual(var.getitem(0), ConstVariable(4))
        var.append(ConstVariable(5))
        self.assertEqual(var.getitem(3), ConstVariable(5))

    def test_delitem(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.delitem(0)
        with self.assertRaises(IndexError):
            var.getitem(2)
        var.pop()
        with self.assertRaises(IndexError):
            var.getitem(1)

    def test_insert(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.insert(0, ConstVariable(4))
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [4, 1, 2, 3]],
        )
        var.insert(2, ConstVariable(5))
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [4, 1, 5, 2, 3]],
        )

    def test_append(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.append(ConstVariable(4))
        self.assertEqual(var.getitem(3), ConstVariable(4))

    def test_extend(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.extend([ConstVariable(4), ConstVariable(5)])
        self.assertEqual(var.getitem(3), ConstVariable(4))
        self.assertEqual(var.getitem(4), ConstVariable(5))

    def test_pop(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        self.assertEqual(var.pop(), ConstVariable(3))
        self.assertEqual(var.pop(0), ConstVariable(1))

    def test_clear(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.clear()
        self.assertEqual(var.proxy.length, 0)

    def test_remove(self):
        data = [1, 2, 3]
        var = ListVariable(data)
        var.remove(ConstVariable(2))
        self.assertEqual(var.getitem(0), ConstVariable(1))
        self.assertEqual(var.getitem(1), ConstVariable(3))

    def test_sort(self):
        data = [2, 3, 0, 4, 1, 5]
        var = ListVariable(data)
        var.sort()
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [0, 1, 2, 3, 4, 5]],
        )

    def test_sort_with_key(self):
        data = [-1, -4, 2, 0, 5, -3]
        var = ListVariable(data)
        var.sort(key=lambda x: x**2)
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [0, -1, 2, -3, -4, 5]],
        )

    def test_sort_reverse(self):
        data = [2, 3, 0, 4, 1, 5]
        var = ListVariable(data)
        var.sort(reverse=True)
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [5, 4, 3, 2, 1, 0]],
        )

    def test_reverse(self):
        data = [2, 3, 0, 4, 1, 5]
        var = ListVariable(data)
        var.reverse()
        print(var.proxy.get_all())
        self.assertEqual(
            [var.getitem(i) for i in range(var.proxy.length)],
            [ConstVariable(n) for n in [5, 1, 4, 0, 3, 2]],
        )


if __name__ == "__main__":
    unittest.main()
