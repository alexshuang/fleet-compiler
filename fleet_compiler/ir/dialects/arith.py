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


class AddIOp(_BinaryOp): ...


class AddFOp(_BinaryOp): ...


class SubIOp(_BinaryOp): ...


class SubFOp(_BinaryOp): ...


class MulIOp(_BinaryOp): ...


class MulFOp(_BinaryOp): ...


class DivSIOp(_BinaryOp): ...


class DivFOp(_BinaryOp): ...
