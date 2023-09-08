import traceback


class SotErrorBase(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ..opcode_translator.breakpoint import BreakpointManager

        BreakpointManager().on_event(f"{self.__class__.__name__}")

    def print(self):
        lines = traceback.format_tb(self.__traceback__)
        print("".join(lines))


class InnerError(SotErrorBase):
    pass


class HasNoAttributeError(InnerError):
    pass


class FallbackError(SotErrorBase):
    def __init__(self, msg, disable_eval_frame=False):
        super().__init__(msg)
        self.disable_eval_frame = False


# raise in inline function call strategy.
class BreakGraphError(SotErrorBase):
    pass


def inner_error_default_handler(func, message_fn):
    """Wrap function and an error handling function and throw an InnerError."""

    def impl(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            message = message_fn(*args, **kwargs)
            origin_exception_message = "\n".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            raise InnerError(
                f"{message}.\nOrigin Exception is: \n {origin_exception_message}"
            ) from e

    return impl
