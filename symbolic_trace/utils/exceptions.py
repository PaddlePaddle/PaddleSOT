class FallbackErrorBase(Exception):
    pass


class InnerError(FallbackErrorBase):
    pass


class UnsupportError(FallbackErrorBase):
    pass


# raise in inline function call strategy.
class BreakGraphError(FallbackErrorBase):
    pass
