from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from fleet_compiler.ir.core import Attribute, Block, BlockArgument, IRType, OpResult, OpTrait, Region
from ..core import *
from .builtin import *
from ..printer import Printer


@dataclass
class FunctionType(IRType):
    input_types: list[IRType]
    output_types: list[IRType]


@dataclass
class FlatSymbolRefAttr(Attribute):
    sym_name: StringAttr


class FuncOp(Operation):
    hasCustomAssemblyFormat = True

    def __init__(self, sym_name: str, function_type: FunctionType, body: Region,
                 arg_attrs: list[tuple] = [], visibility: str = 'public'):
        self.function_type = function_type
        super().__init__(result_types=function_type.output_types,
                         regions=[body],
                         attributes={
                             'sym_name': StringAttr(sym_name),
                             'visibility': StringAttr(visibility),
                             'function_type': function_type,
                             'arg_attrs': arg_attrs,
                         })
    
    def print(self, p: Printer):
        sym_name = self.attributes['sym_name'].value
        prefix = p._get_indent()
        func_type = self.attributes['function_type']
        input_types = [p._get_type_str(o) for o in func_type.input_types]
        output_types = [p._get_type_str(o) for o in func_type.output_types]
        arg_str = ', '.join([f'%arg{i}: {o}' for i, o in enumerate(input_types)])
        output_type_str = ', '.join(output_types)
        p._print_string(f"{prefix}{self.name} @{sym_name}({arg_str}) -> ({output_type_str}) ")
        p._print_region(self.regions[0])


class ReturnOp(Operation):
    hasCustomAssemblyFormat = True

    def __init__(self, operands: list[Value]):
        super().__init__(operands=operands)

    def print(self, p: Printer):
        prefix = p._get_indent()
        if len(self.operands) > 0:
            rets = ','.join([o.name for o in self.operands])
            ret_types = ','.join([p._get_type_str(o.type) for o in self.operands])
            ret = f' {rets}: {ret_types}'
        else:
            ret = ''
        p._print_string(f"{prefix}return{ret}\n")


class CallOp(Operation):
    hasCustomAssemblyFormat = True

    def __init__(self, operands: list[Value], result_types: list[IRType],
                 attributes: dict[str, Attribute]):
        super().__init__(operands=operands, result_types=result_types,
                         attributes=attributes)

    def print(self, p: Printer):
        prefix = p._get_indent()
        results = ','.join([o.name for o in self.results])
        if results != '':
            results += ' = '
        sym_name = self.attributes['callee'].sym_name.value
        operands = ','.join([o.name for o in self.operands])
        operand_types = ','.join([p._get_type_str(o.type) for o in self.operands])
        output_types = ','.join([p._get_type_str(o.type) for o in self.results])
        p._print_string(f"{prefix}{results}func.call @{sym_name}({operands}) : ({operand_types}) -> ({output_types})\n")
