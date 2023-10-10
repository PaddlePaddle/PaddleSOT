import unittest

from test_case_base import TestCaseBase, strict_mode_guard

import sot


def fn_with_try_except():
    sot.psdb.breakgraph()
    sot.psdb.fallback()
    try:
        raise ValueError("ValueError")
    except ValueError:
        print("catch ValueError")
        return True


class TestErrorHandling(TestCaseBase):
    @strict_mode_guard(0)
    def test_fn_with_try_except(self):
        self.assert_results(fn_with_try_except)


if __name__ == "__main__":
    unittest.main()
