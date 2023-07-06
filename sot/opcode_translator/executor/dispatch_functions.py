# This file stores the customed function that will be called by the dispatch mechanism.

from ...utils import BreakGraphError, NotImplementException


def raise_break_graph_fn(*args, **kwarg):
    raise BreakGraphError("raise by raise_break_graph_fn.")


def raise_not_implement_fn(*args, **kwarg):
    raise NotImplementException("raise by raise_break_graph_fn.")


# just a function for operator.in
def operator_in(left, right):
    return left in right


def operator_not_in(left, right):
    return left not in right


def operator_exception_match(left, right):
    pass


def operator_BAD(left, right):
    pass


def tensor_numel(x):
    pass
