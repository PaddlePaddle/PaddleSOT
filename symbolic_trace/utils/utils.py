import os
import logging
import paddle

class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}
    def __call__(self):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls()
        return self._instance[self._cls]

class NameGenerator:
    def __init__(self, prefix):
        self.counter = 0
        self.prefix = prefix

    def next(self):
        name = self.prefix + str(self.counter)
        self.counter += 1
        return name


def log(level, *args):
    cur_level = int(os.environ.get('LOG_LEVEL', '1'))
    if level <= cur_level:
        print(*args, end="")

def log_do(level, fn):
    cur_level = int(os.environ.get('LOG_LEVEL', '1'))
    if level <= cur_level:
        fn()

def no_eval_frame(func):
    def no_eval_frame_func(*args):
        old_cb = paddle.fluid.core.set_eval_frame(None)
        retval = func(*args)
        paddle.fluid.core.set_eval_frame(old_cb)
        return retval
    return no_eval_frame_func