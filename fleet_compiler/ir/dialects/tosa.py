from __future__ import annotations

from ..core import *
from .builtin import *


class _BinaryOp(Operation):
    def __init__(self, lhs: Value, rhs: Value, attrs: dict[str, Attribute] = {}):
        if lhs.type != rhs.type:
            raise ValueError(f"lhs.type != rhs.type: {lhs.type} vs. {rhs.type}")
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type], attributes=attrs)


class _UnaryOp(Operation):
    def __init__(self, input: Value):
        super().__init__(operands=[input], result_types=[input.type])


class AddOp(_BinaryOp):
    name = 'tosa.add'


class SubOp(_BinaryOp):
    name = 'tosa.sub'


class MulOp(_BinaryOp):
    name = 'tosa.mul'

    def __init__(self, lhs: Value, rhs: Value):
        # not support quantlization
        attrs = {'shift': IntegerAttr(0, IntegerType(32, True))}
        super().__init__(lhs, rhs, attrs)


class ReciprocalOp(_UnaryOp):
    name = 'tosa.reciprocal'