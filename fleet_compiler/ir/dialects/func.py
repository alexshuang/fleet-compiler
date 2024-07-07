from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from fleet_compiler.ir.core import Attribute, Block, BlockArgument, IRType, OpResult, OpTrait, Region
from ..core import *
from .builtin import *


@dataclass
class FunctionType(IRType):
    input_types: list[IRType]
    output_types: list[IRType]


@dataclass
class FlatSymbolRefAttr(Attribute):
    sym_name: StringAttr


class FuncOp(Operation):
    name = 'func.func'

    def __init__(self, sym_name: str, function_type: FunctionType,
                 operands: list[Value], visibility: str = 'public'):
        self.function_type = function_type
        super().__init__(operands=operands, result_types=function_type.output_types,
                         regions=[Region([Block()])],
                         attributes={
                             'sym_name': StringAttr(sym_name),
                             'visibility': StringAttr(visibility),
                             'function_type': function_type,
                         })


class ReturnOp(Operation):
    name = 'func.return'

    def __init__(self, operands: list[Value]):
        super().__init__(operands=operands)


class CallOp(Operation):
    name = 'func.call'

    def __init__(self, operands: list[Value], result_types: list[IRType],
                 attributes: dict[str, Attribute]):
        super().__init__(operands=operands, result_types=result_types,
                         attributes=attributes)
