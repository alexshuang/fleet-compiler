from __future__ import annotations

from ..core import *


class AddOp(Operation):
    def __init__(self, lhs: Value, rhs: Value):
        self.name = 'tosa.add'
        if lhs.type != rhs.type:
            raise ValueError(f"lhs.type != rhs.type: {lhs.type} vs. {rhs.type}")
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])
