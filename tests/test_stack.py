import unittest

from sot.opcode_translator.executor.variable_stack import VariableStack


class TestVariableStack(unittest.TestCase):
    def test_basic(self):
        stack = VariableStack([1, 2, 3])
        self.assertEqual(str(stack), "[1, 2, 3]")
        self.assertEqual(len(stack), 3)
        self.assertEqual(str(stack.copy()), str(stack))

    def test_peek(self):
        stack = VariableStack([1, 2, 3])
        self.assertEqual(stack.peek(), 3)
        self.assertEqual(stack.top, 3)
        self.assertEqual(stack.peek(1), 3)
        self.assertEqual(stack.peek[1], 3)
        self.assertEqual(stack.peek[:1], [3])
        self.assertEqual(stack.peek[:2], [2, 3])

    def test_push_pop(self):
        stack = VariableStack()
        stack.push(1)
        stack.push(2)
        self.assertEqual(stack.pop(), 2)
        self.assertEqual(stack.pop(), 1)

    def test_pop_n(self):
        stack = VariableStack([1, 2, 3, 4])
        self.assertEqual(stack.pop_n(2), [3, 4])
        self.assertEqual(stack.pop_n(2), [1, 2])


if __name__ == "__main__":
    unittest.main()
