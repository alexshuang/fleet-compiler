from __future__ import annotations

from typing import Union
from ..core import *
from .builtin import *


class SplatOp(Operation):
    def __init__(self, input: Value, output_type: Union[RankedTensorType | UnrankedTensorType]):
        super().__init__(operands=[input], result_types=[output_type])
