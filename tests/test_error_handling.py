import unittest

import sot


def fn_with_try_except():
    sot.psdb.breakgraph()
    sot.psdb.fallback()
    try:
        raise ValueError("ValueError")
    except ValueError:
        print("catch ValueError")


class TestErrorHandling(unittest.TestCase):
    def test_fn_with_try_except(self):
        sot.symbolic_translate(fn_with_try_except)()


if __name__ == "__main__":
    unittest.main()
