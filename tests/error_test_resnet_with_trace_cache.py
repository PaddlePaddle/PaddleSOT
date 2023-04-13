import os
os.environ['FLAGS_cudnn_deterministic'] = "True"

import paddle
import numpy as np
import random
from numpy.testing import assert_array_equal
from symbolic_trace import symbolic_trace
import unittest
from paddle.vision import resnet50
from symbolic_trace.utils.utils import execute_time
from symbolic_trace.symbolic.compile_cache import CompileSIRCache
from symbolic_trace.trace_cache_entrance import trace_cache
from types import MethodType

@trace_cache
def forward_with_cache(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    if self.with_pool:
        x = self.avgpool(x)
    if self.num_classes > 0:
        x = paddle.flatten(x, 1)
        x = self.fc(x)
    return x

def run_dygraph_optimizer(inp, to_static):
    """ dygraph train + SGD optimizer
    """
    paddle.seed(2021)
    np.random.seed(2021)
    random.seed(2021)
    net = resnet50()
    if to_static:
        net.forward = MethodType(forward_with_cache, net)
        net.forward = symbolic_trace(net.forward)
        #net = paddle.jit.to_static(net)
    optimizer = paddle.optimizer.SGD(learning_rate=0.03,
        parameters=net.parameters())
    for i in range(5):
        optimizer.clear_grad()
        loss = execute_time(net)(inp)
        loss.backward()
        optimizer.step()
        print("===============================================")
    return loss

class TestBackward(unittest.TestCase):
    def test(self):
        #TODO(xiongkun) add cache to speedup !
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        inp = paddle.rand((3, 3, 255, 255))
        out1 = run_dygraph_optimizer(inp, True )[0].numpy()
        out2 = run_dygraph_optimizer(inp, False)[0].numpy()
        assert_array_equal(out1, out2, "Not Equal in dygraph and static graph", True)
        assert CompileSIRCache().hit_num == 4, f"CompileSIRCache hit_num should be 4, but{CompileSIRCache().hit_num}"

if __name__ == "__main__":
    unittest.main()
