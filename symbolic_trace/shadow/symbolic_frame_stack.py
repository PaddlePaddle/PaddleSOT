from typing import Optional

def top():
    return _current_frame

def push(frame: Optional["SymbolicFrame"]):
    assert frame.f_back == _current_frame
    _update_current_frame(frame)

def pop(frame: Optional["SymbolicFrame"]):
    assert _current_frame.f_back == frame
    _update_current_frame(frame)

def _update_current_frame(frame: Optional["SymbolicFrame"]):
    global _current_frame
    _current_frame = frame

_current_frame: "SymbolicFrame" = None
