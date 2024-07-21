from __future__ import annotations

from ..core import *
from .builtin import *


class PrintOp(Operation):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        super().__init__(operands=args)
