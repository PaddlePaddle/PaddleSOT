import traceback


class FallbackErrorBase(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ..opcode_translator.breakpoint import BreakpointManager

        BreakpointManager().on_event(f"{self.__class__.__name__}")

    def print(self):
        lines = traceback.format_tb(self.__traceback__)
        print("".join(lines))


class InnerError(FallbackErrorBase):
    pass


class NotImplementException(FallbackErrorBase):
    pass


# raise in inline function call strategy.
class BreakGraphError(FallbackErrorBase):
    pass


def inner_error_default_handler(func, message_fn):
    """Wrap function and an error handling function and throw an InnerError."""

    def impl(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            message = message_fn(*args, **kwargs)
            raise InnerError(
                f"{message}.\nOrigin Exception is : \n {traceback.format_exception(type(e), e, e.__traceback__)}"
            ) from e

    return impl
