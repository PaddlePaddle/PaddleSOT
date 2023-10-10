from __future__ import annotations

import types
from typing import NamedTuple


class CustomCode(NamedTuple):
    code: types.CodeType | None
    disable_eval_frame: bool
