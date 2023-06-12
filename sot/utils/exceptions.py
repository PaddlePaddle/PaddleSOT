class FallbackErrorBase(Exception):
    pass


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
            raise InnerError(f"{message}.\nOrigin Exception is : \n {e}")

    return impl
