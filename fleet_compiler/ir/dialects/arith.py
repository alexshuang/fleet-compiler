from __future__ import annotations

from ..core import *


class ConstantOp(Operation):
    def __init__(self, attr: Attribute, type: IRType):
        self.name = 'arith.constant'
        super().__init__(result_types=[type], attributes={'value': attr})


class AddIOp(Operation):
    def __init__(self, lhs: Value, rhs: Value):
        self.name = 'arith.addi'
        if lhs.type != rhs.type:
            raise ValueError(f"lhs.type != rhs.type: {lhs.type} vs. {rhs.type}")
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])


class AddFOp(Operation):
    def __init__(self, lhs: Value, rhs: Value):
        self.name = 'arith.addf'
        if lhs.type != rhs.type:
            raise ValueError(f"lhs.type != rhs.type: {lhs.type} vs. {rhs.type}")
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])
