import os
import unittest

import numpy as np
from test_case_base import TestCaseBase

import paddle
from sot.psdb import check_no_breakgraph, check_no_fallback

os.environ['MIN_GRAPH_SIZE'] = '-1'


@check_no_breakgraph
@check_no_fallback
def forward(x, y):
    if x == 0:
        return y + 2
    else:
        return y * 2


@check_no_breakgraph
@check_no_fallback
def forward2(x, y):
    if x == x:  # numpy == numpy
        return y + 2
    else:
        return y * 2


class TestJumpWithNumpy(TestCaseBase):
    def test_jump(self):
        self.assert_results(forward, np.array([1]), paddle.to_tensor(2))
        self.assert_results(forward, np.array([0]), paddle.to_tensor(2))
        self.assert_results(forward2, np.array([0]), paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()
