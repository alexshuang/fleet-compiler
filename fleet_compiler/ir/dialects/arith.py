from __future__ import annotations

from ..core import *
from .builtin import *
from ..traits import *


class ConstantOp(Operation):
    def __init__(self, attr: Attribute, type: IRType):
        super().__init__(result_types=[type], attributes={'value': attr}, traits=[Pure()])


class _BinaryOp(Operation):
    def __init__(self, lhs: Value, rhs: Value):
        output_type = lhs.type
        if lhs.type != rhs.type:
            lhs_perf = get_preference_by_type_cls(lhs.type)
            rhs_perf = get_preference_by_type_cls(rhs.type)
            output_type = lhs.type if lhs_perf > rhs_perf else rhs.type
        super().__init__(operands=[lhs, rhs], result_types=[output_type])


class AddIOp(_BinaryOp): ...


class AddFOp(_BinaryOp): ...


class SubIOp(_BinaryOp): ...


class SubFOp(_BinaryOp): ...


class MulIOp(_BinaryOp): ...


class MulFOp(_BinaryOp): ...


class DivSIOp(_BinaryOp): ...


class DivFOp(_BinaryOp): ...
