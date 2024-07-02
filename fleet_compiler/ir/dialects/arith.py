from __future__ import annotations

from ..core import *


class ConstantOp(Operation):
    def __init__(self, attr: Attribute, type: IRType):
        self.name = 'arith.constant'
        super().__init__(result_types=[type], attributes={'value': attr})


class _BinaryOp(Operation):
    def __init__(self, lhs: Value, rhs: Value):
        if lhs.type != rhs.type:
            raise ValueError(f"lhs.type != rhs.type: {lhs.type} vs. {rhs.type}")
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])


class AddIOp(_BinaryOp):
    name = 'arith.addi'


class AddFOp(_BinaryOp):
    name = 'arith.addf'


class SubIOp(_BinaryOp):
    name = 'arith.subi'


class SubFOp(_BinaryOp):
    name = 'arith.subf'


class MulIOp(_BinaryOp):
    name = 'arith.muli'


class MulFOp(_BinaryOp):
    name = 'arith.mulf'


class DivSIOp(_BinaryOp):
    name = 'arith.divsi'


class DivFOp(_BinaryOp):
    name = 'arith.divf'
