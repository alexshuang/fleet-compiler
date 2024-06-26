from __future__ import annotations

from ..core import *


class Constant(Operation):
    def __init__(self, attr: Attribute, type: IRType):
        self.name = 'arith.constant'
        super().__init__(result_types=[type], attributes={'value': attr})
