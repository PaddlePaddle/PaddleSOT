import os
import logging
import paddle
from .paddle_api_config import paddle_api_list, fallback_list, paddle_api_module_prefix
from paddle.utils import map_structure, flatten
from frozendict import frozendict
import functools
import types
import time
from weakref import WeakValueDictionary  

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
    cur_level = int(os.environ.get('LOG_LEVEL', '0'))
    if level <= cur_level:
        print(*args, end="")

def log_do(level, fn):
    cur_level = int(os.environ.get('LOG_LEVEL', '0'))
    if level <= cur_level:
        fn()

def no_eval_frame(func):
    def no_eval_frame_func(*args, **kwargs):
        old_cb = paddle.fluid.core.set_eval_frame(None)
        try:
            retval = func(*args, **kwargs)
        except:
            raise
        finally:
            paddle.fluid.core.set_eval_frame(old_cb)
        return retval
    return no_eval_frame_func

def is_paddle_api(func):
    if isinstance(func, paddle.nn.Layer):  # ignore all the classes
        return False
    if hasattr(func, "__self__"): #ignore all the methods
        return False
    return in_paddle_module(func) or func in paddle_api_list

def in_paddle_module(func):
    if hasattr(func, '__module__'): 
        module_str = func.__module__
        log(5, "find paddle function with __module__: ", module_str, "\n")
        log(5, "                     with __name__  : ", func.__name__, "\n")
        log(5, "                     with results   : ")
        for prefix in paddle_api_module_prefix:
            if module_str.startswith(prefix):
                log(5, " True\n")
                return True
    log(5, " False\n")
    return False

def is_fallback_api(func):
    return func in fallback_list

def is_proxy_tensor(obj):
    return hasattr(obj, '_proxy_tensor_')

def map_if(*structures, pred, true_fn, false_fn): 
    def replace(*args):
        if pred(*args):
            return true_fn(*args)
        return false_fn(*args)
    return map_structure(replace, *structures)

def count_if(*structures, pred):
    def is_true(*args):
        if pred(*args):
            return 1
        return 0
    return sum(flatten(map_structure(is_true, *structures)))

def freeze_structure(structure):
    """
    only support list / dict and its nested structure
    """
    if isinstance(structure, (list, tuple)):
        return tuple(map(freeze_structure, structure))
    if isinstance(structure, dict):
        return frozendict({k: freeze_structure(v) for k, v in structure.items()})
    #if isinstance(structure, types.CodeType): 
        #return id(structure)
    return structure

class Cache:
    def __init__(self, weak=False):
        if not weak: 
            self.cache = {}
        else: 
            self.cache = WeakValueDictionary()
        self.hit_num = 0

    def __call__(self, *args, **kwargs): 
        cache_key = self.key_fn(*args, **kwargs)
        if cache_key in self.cache:
            log(5, "cache hit: ", cache_key, "\n")
            self.hit_num += 1
            return self.cache[cache_key]
        value = self.value_fn(*args, **kwargs)
        self.cache[cache_key] = value
        return value

    def clear(self):
        self.cache.clear()
        self.hit_num = 0

    def key_fn(self, *args, **kwargs):
        raise NotImplementedError()

    def value_fn(self, *args, **kwargs):
        raise NotImplementedError()

def execute_time(func):  
    def wrapper(*args, **kwargs):  
        start_time = time.time()  
        result = func(*args, **kwargs)  
        end_time = time.time()  
        execution_time = end_time - start_time  
        print("Execute time:", execution_time)
        return result  
    return wrapper

def meta_str(shape, dtype, stop_gradient):
    return f"(shape: {shape}, dtype: {dtype}, stop_gradient: {stop_gradient})"