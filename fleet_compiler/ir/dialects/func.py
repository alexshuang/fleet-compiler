from __future__ import annotations

from dataclasses import dataclass, field
from ..core import *
from .builtin import *


@dataclass
class FunctionType(IRType):
    input_types: list[IRType]
    output_types: list[IRType]


class FuncOp(Operation):
    name = 'func.func'

    def __init__(self, sym_name: str, function_type: FunctionType,
                 operands: list[Value], visibility: str = 'public'):
        super().__init__(operands=operands, result_types=function_type.output_types,
                         regions=[Region([Block()])],
                         attributes={
                             'sym_name': StringAttr(sym_name),
                             'visibility': StringAttr(visibility),
                             'function_type': function_type,
                         })
