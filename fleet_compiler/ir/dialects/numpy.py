from __future__ import annotations

from ..core import *
from .builtin import *


class Random_RandnOp(Operation):
    def __init__(self, operands: list[Value]):
        try:
            output_shape = [o.owner().attributes['value'].value for o in operands]
        except:
            raise ValueError("Expect all operands are constant op")
        type = RankedTensorType(output_shape, FloatType(32))
        super().__init__(operands=operands, result_types=[type])
