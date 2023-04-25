from .code_gen import gen_instr


def from_instruction(instr):
    pass


class Source:
    def gen_instructions(self):
        raise NotImplementedError()

    def gen_guard(self, value):
        raise NotImplementedError()


class LocalSource(Source):
    def __init__(self, idx, name):
        super().__init__()
        self.name = name
        self.idx = idx

    def gen_instructions(self):
        return [gen_instr("LOAD_FAST", self.idx, self.name)]

    def gen_guard_fn(self, value):
        guard_fn = lambda frame: frame.f_locals[self.name] == value
        return guard_fn


class GlobalSource(Source):
    def __init__(self, name):
        super().__init__()
        self.name = name


class GetAttrSource(Source):
    def __init__(self, obj_source, attr):
        super().__init__()
        self.attr = attr
        self.obj = obj_source


class GetItemSource(Source):
    def __init__(self, obj_source, attr):
        super().__init__()
        self.attr = attr
        self.obj = obj_source
