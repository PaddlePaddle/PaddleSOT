from __future__ import annotations

import unittest

from test_case_base import TestCaseBase


# copy from python library _distutils_hack/__init__.py
def find_spec(self, fullname, path, target=None):
    method_name = 'spec_for_{fullname}'.format(**locals())
    method = getattr(self, method_name, lambda: None)
    return method()


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(find_spec, "self", "fullname", "path", None)


if __name__ == "__main__":
    unittest.main()
