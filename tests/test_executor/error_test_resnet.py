import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.vision import resnet18


def resnet_call(x: paddle.Tensor, net: paddle.nn.Layer):
    return net(x)


class TestLayer(TestCaseBase):
    def test_layer(self):
        x = paddle.rand((10, 3, 224, 224))
        net = resnet18(pretrained=False)
        net.eval()
        self.assert_results(resnet_call, x, net)


if __name__ == "__main__":
    unittest.main()
