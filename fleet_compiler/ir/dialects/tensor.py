from __future__ import annotations

from typing import Union
from ..core import *
from .builtin import *
from ..interfaces import ShapeInferenceOpInterface


class SplatOp(Operation):
    def __init__(self, input: Value, output_type: Union[RankedTensorType | UnrankedTensorType]):
        super().__init__(operands=[input], result_types=[output_type])


class CastOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, input: Value, output_type: RankedTensorType | UnrankedTensorType):
        super().__init__(operands=[input], result_types=[output_type])

    def infer_shapes(self, op: Operation):
        if isinstance(in_type := op.operands[0].type, UnrankedTensorType):
            raise ValueError(f"Invalid operand type: {in_type}")
        op.results[0].type = in_type