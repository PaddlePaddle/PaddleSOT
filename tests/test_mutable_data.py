import unittest

from sot.opcode_translator.executor.mutable_data import (
    MutableData,
    MutableDictLikeData,
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


if __name__ == "__main__":
    unittest.main()
