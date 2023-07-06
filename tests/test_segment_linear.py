import unittest

from test_case_base import TestCaseBase

import paddle
import sot
from paddle import nn


class Head(nn.Layer):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(10, 150)

    def forward(self, x, patch_embed_size):
        masks = self.head(x)
        # [b, (h w), c] -> [b, c, h, w]
        h, w = patch_embed_size[0], patch_embed_size[1]
        masks = masks.reshape((1, h, w, paddle.shape(masks)[-1]))
        masks = masks.transpose((0, 3, 1, 2))
        return masks


class SimpleNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.tmp = nn.Linear(1, 1024 * 10)
        self.tmp2 = nn.Linear(1, 1 * 10 * 32 * 32)
        self.head = Head()

    def getshape(self, x):
        x = self.tmp2(x.mean().reshape([1])).reshape([1, 10, 32, 32])
        x = paddle.shape(x)
        return x

    def forward(self, x):
        shape = self.getshape(x)
        feat = self.tmp(x.mean().reshape([1])).reshape([1, 1024, 10])
        logits = self.head(feat, shape[2:])
        return logits


class TestExecutor(TestCaseBase):
    def test_simple(self):
        sot.skip_function(SimpleNet.forward)
        x = paddle.randn((1, 8, 8))
        net = SimpleNet()
        net = paddle.jit.to_static(net)
        loss = net(x)
        loss = loss.sum()
        loss.backward()


if __name__ == "__main__":
    unittest.main()
