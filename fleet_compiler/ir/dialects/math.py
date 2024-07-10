from __future__ import annotations

from ..core import *
from .builtin import *


class FPowIOp(Operation):
    def __init__(self, lhs: Value, rhs: Value, attrs: dict[str, Attribute] = {}):
        lhs_type, rhs_type = lhs.type, rhs.type
        if isinstance(lhs_type, RankedTensorType):
            if not isinstance(rhs_type, RankedTensorType):
                raise ValueError(f"Invalid lhs/rhs to make power op! {lhs.type} vs. {rhs.type}")
            lhs_type, rhs_type = lhs_type.element_type, rhs_type.element_type

        if not isinstance(lhs_type, FloatType) or not isinstance(rhs_type, IntegerType):
            raise ValueError(f"invalid lhs/rhs: {lhs_type}, {rhs_type} for math.fpowi")
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type], attributes=attrs)


class IPowIOp(Operation):
    def __init__(self, lhs: Value, rhs: Value, attrs: dict[str, Attribute] = {}):
        lhs_type, rhs_type = lhs.type, rhs.type
        if isinstance(lhs_type, RankedTensorType):
            if not isinstance(rhs_type, RankedTensorType):
                raise ValueError(f"Invalid lhs/rhs to make power op! {lhs.type} vs. {rhs.type}")
            lhs_type, rhs_type = lhs_type.element_type, rhs_type.element_type

        if lhs_type != rhs_type or not isinstance(lhs_type, IntegerType):
            raise ValueError(f"invalid lhs/rhs: {lhs_type}, {rhs_type} for math.ipowi")
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type], attributes=attrs)


class PowFOp(Operation):
    def __init__(self, lhs: Value, rhs: Value, attrs: dict[str, Attribute] = {}):
        lhs_type, rhs_type = lhs.type, rhs.type
        if isinstance(lhs_type, RankedTensorType):
            if not isinstance(rhs_type, RankedTensorType):
                raise ValueError(f"Invalid lhs/rhs to make power op! {lhs.type} vs. {rhs.type}")
            lhs_type, rhs_type = lhs_type.element_type, rhs_type.element_type

        if lhs_type != rhs_type or not isinstance(lhs_type, FloatType):
            raise ValueError(f"invalid lhs/rhs: {lhs_type}, {rhs_type} for math.powf")
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type], attributes=attrs)
