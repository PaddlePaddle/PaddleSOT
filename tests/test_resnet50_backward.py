import os
os.environ['FLAGS_cudnn_deterministic'] = "True"

import paddle
import numpy as np
import random
from numpy.testing import assert_array_equal
from symbolic_trace import symbolic_trace
import unittest
from paddle.vision import resnet50

def run_dygraph_optimizer(inp, to_static):
    """ dygraph train + SGD optimizer
    """
    paddle.seed(2021)
    np.random.seed(2021)
    random.seed(2021)
    net = resnet50()
    if to_static:
        net.forward = symbolic_trace(net.forward)
    optimizer = paddle.optimizer.SGD(learning_rate=0.03,
        parameters=net.parameters())
    for i in range(5):
        optimizer.clear_grad()
        loss = net(inp)
        loss.backward()
        optimizer.step()
    return loss
    
class TestBackward(unittest.TestCase):
    def test(self):
        #TODO(xiongkun) add cache to speedup !
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        inp = paddle.rand((3, 3, 255, 255))
        assert_array_equal(run_dygraph_optimizer(inp, True)[0].numpy(), run_dygraph_optimizer(inp, False)[0].numpy(), "Not Equal in dygraph and static graph", True)

if __name__ == "__main__":
    unittest.main()
