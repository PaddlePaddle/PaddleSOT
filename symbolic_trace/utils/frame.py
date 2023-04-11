import inspect

from .utils import log

def find_user_defined_func_frames(frame_start_func, frame_end_funcs):
    # TODO(SigureMo): Find a better way to automatically get the calling frame
    current_frame = inspect.currentframe()
    assert current_frame is not None
    calling_frame = current_frame

    # Record all calling frames
    calling_stack = []
    while calling_frame.f_back is not None:
        calling_stack.append((calling_frame.f_code.co_name, calling_frame))
        calling_frame = calling_frame.f_back

    calling_stack = list(reversed(calling_stack))

    # Analysis which frame is user defined function
    # The calling_stack like this:
    # func1 -> func2 -> func3 -> symbolic_traced_func -> user_func1 -> user_func2 -> no_eval_frame_func
    #       -> symbolic_inner_func_0 -> no_eval_frame_func -> symbolic_inner_func_1 -> ...
    # We need to find the frame of user_func1 and user_func2.
    frame_start_idx = 0
    frame_end_idx = len(calling_stack) - 1
    for frame_idx, (frame_name, _) in enumerate(calling_stack):
        if frame_name == frame_start_func:
            frame_start_idx = frame_idx + 1
        if frame_name in frame_end_funcs:
            frame_end_idx = frame_idx
            break
    
    assert frame_end_idx != len(calling_stack) - 1, "Can not find no_eval_frame_func in calling stack."

    log(5, "Found user defined frame", calling_stack[frame_end_idx - 1][0])
    calling_frames = list(reversed([frame for _, frame in calling_stack[frame_start_idx: frame_end_idx]]))
    return calling_frames
