from ..instruction_utils import gen_instr


def from_instruction(instr):
    pass


class Tracker:
    def __init__(self, children=[]):
        self.children = children

    def is_leaf(self):
        return len(self.children) == 0

    def gen_instructions(self):
        raise NotImplementedError()

    def gen_guard(self, value):
        raise NotImplementedError()

    def trace_value_from_frame(self):
        raise NotImplementedError()

    def gen_guard_fn(self, value):
        guard_fn = lambda frame: self.trace_value_from_frame()(frame) == value
        return guard_fn


class LocalTracker(Tracker):
    def __init__(self, idx, name):
        super().__init__()
        self.name = name
        self.idx = idx

    def gen_instructions(self):
        return [gen_instr("LOAD_FAST", self.idx, self.name)]

    def trace_value_from_frame(self):
        return lambda frame: frame.f_locals[self.name]


class GlobalTracker(Tracker):
    def __init__(self, name):
        super().__init__()
        self.name = name


class GetAttrTracker(Tracker):
    def __init__(self, obj_source, attr):
        super().__init__()
        self.attr = attr
        self.obj = obj_source


class GetItemTracker(Tracker):
    def __init__(self, container, key):
        super().__init__()
        self.key = key
        self.container = container

    def gen_instructions(self):
        return (
            self.struct.source.gen_instructions()
            + self.key.source.gen_instructions()
            + [gen_instr("BINARY_SUBSCR", 0, 0)]
        )

    def trace_value_from_frame(self):
        def trace_value(frame):
            container = self.container.trace_value_from_frame()(frame)
            key = self.key.trace_value_from_frame()(frame)
            return container[key]

        return trace_value
