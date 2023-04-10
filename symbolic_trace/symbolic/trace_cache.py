import paddle

from ..utils import Singleton, is_proxy_tensor, log

@Singleton
class TraceCache():
    def __init__(self):
        self.cache = {}
        self.hit_num = 0

    def key_fn(self, name, inputs):
        # need a better hash strategy
        key_list = [name]
        for inp in paddle.utils.flatten(inputs):
            if is_proxy_tensor(inp):
                key_list.append(str(inp.meta))
            else:
                key_list.append(inp)

        key = hash(tuple(key_list))
        return key

    def hit(self, key):
        if key in self.cache.keys():
            log(5, "cache hit: ", key, "\n")
            self.hit_num += 1
            return True
        else:
            return False

    def clear(self):
        self.cache.clear()
        self.hit_num = 0

    def set_key_value(self, key, value):
        self.cache[key] = value
    
    def get_value(self, key):
        return self.cache[key]