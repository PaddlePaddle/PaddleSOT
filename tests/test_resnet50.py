import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.vision import resnet50


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        return x


class TestNet(TestCaseBase):
    def test(self):
        image = paddle.rand((1, 3, 255, 255))
        net = resnet50()
        self.assert_results(net, image)


if __name__ == "__main__":
    unittest.main()
