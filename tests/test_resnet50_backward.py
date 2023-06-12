import os

os.environ["FLAGS_cudnn_deterministic"] = "True"

import random
import unittest

import numpy as np
from numpy.testing import assert_array_equal

import paddle
from paddle.vision import resnet50
from sot import symbolic_translate
from sot.utils.utils import execute_time


def resnet_call(net: paddle.nn.Layer, x: paddle.Tensor):
    return net(x)


def run_dygraph_optimizer(inp):
    """dygraph train + SGD optimizer"""
    paddle.seed(2021)
    np.random.seed(2021)
    random.seed(2021)
    net = resnet50()
    optimizer = paddle.optimizer.SGD(
        learning_rate=0.03, parameters=net.parameters()
    )
    for i in range(5):
        optimizer.clear_grad()
        loss = execute_time(net)(inp)
        loss.backward()
        optimizer.step()
    return loss


def run_symbolic_optimizer(inp):
    """dygraph train + SGD optimizer"""
    paddle.seed(2021)
    np.random.seed(2021)
    random.seed(2021)
    net = resnet50()
    net_wrapper = symbolic_translate(resnet_call)
    optimizer = paddle.optimizer.SGD(
        learning_rate=0.03, parameters=net.parameters()
    )
    for i in range(5):
        optimizer.clear_grad()
        loss = execute_time(net_wrapper)(net, inp)
        loss.backward()
        optimizer.step()
    return loss


def run_to_static_optimizer(inp):
    """dygraph train + SGD optimizer"""
    paddle.seed(2021)
    np.random.seed(2021)
    random.seed(2021)
    net = resnet50()
    net = paddle.jit.to_static(net, enable_fallback=False)
    optimizer = paddle.optimizer.SGD(
        learning_rate=0.03, parameters=net.parameters()
    )
    for i in range(5):
        optimizer.clear_grad()
        loss = execute_time(net)(inp)
        loss.backward()
        optimizer.step()
    return loss


class TestBackward(unittest.TestCase):
    def test(self):
        # TODO(xiongkun) add cache to speedup !
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        inp = paddle.rand((3, 3, 255, 255))
        print("Start Run SymbolicTranslate:")
        out2 = run_symbolic_optimizer(inp)[0].numpy()
        print("Start Run Dygraph:")
        out1 = run_dygraph_optimizer(inp)[0].numpy()
        print("Start Run To Static:")
        out1 = run_to_static_optimizer(inp)[0].numpy()
        assert_array_equal(
            out1, out2, "Not Equal in dygraph and static graph", True
        )


if __name__ == "__main__":
    unittest.main()
