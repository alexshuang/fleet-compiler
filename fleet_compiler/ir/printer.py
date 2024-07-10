# ===- printer.py -------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

from .core import *
from .builder import *
from .dialects.builtin import *
from .dialects.func import *

from dataclasses import dataclass, field


def has_custom_print(op: Operation):
    return hasattr(op, 'print') and callable(getattr(op, 'print'))


@dataclass
class Printer:
    _indent: int = field(default=0)
    _indent_space_sizes: int = field(default=2)

    def print(self, op: Operation):
        if isinstance(op, Operation):
            if has_custom_print(op):
                op.print(self._get_indent())
            else:
                if isinstance(op, FuncOp):
                    self._print_func(op)
                elif isinstance(op, ReturnOp):
                    self._print_ret(op)
                elif isinstance(op, CallOp):
                    self._print_call(op)
                else:
                    self._print_results(op)
                    self._print_op_with_default_format(op)
        elif isinstance(op, Region):
            self._print_region(op)
        elif isinstance(op, Block):
            self._print_block(op)
        elif isinstance(op, Value):
            self._print_string(op.name)

    def _get_indent(self):
        return " " * self._indent_space_sizes * self._indent

    def _print_results(self, op: Operation):
        self._print_string(self._get_indent())
        res = op.results
        if len(res) == 0:
            return
        self.print_list(res, self.print)
        self._print_string(" = ")
    
    def print_list(self, elems: Sequence[Operation], func, sep=','):
        for i, elem in enumerate(elems):
            if i:
                self._print_string(sep)
            func(elem)

    def print_dict(self, data: dict[str, Attribute], func, sep=','):
        for i, (k, v) in enumerate(data.items()):
            if i:
                self._print_string(sep)
            func(k, v)

    def _print_string(self, text: str):
        print(text, end="")
    
    def _print_new_line(self):
        print('')

    def _print_op_with_default_format(self, op: Operation):
        self._print_string(f"\"{op.name}\"")
        self._print_operands(op)
        self._print_regions(op)
        self._print_attributes(op)
        self._print_string(" : ")
        self._print_op_type(op)
        self._print_new_line()

    def _print_operands(self, op: Operation):
        self._print_string(" (")
        self.print_list(op.operands, self._print_operand)
        self._print_string(")")
    
    def _print_operand(self, operand: Value):
        self._print_string(operand.name)

    def _print_regions(self, op: Operation):
        if len(op.regions) > 0 and len(op.regions[0].blocks):
            self._print_string(" (")
            self.print_list(op.regions, self._print_region)
            if self._indent > 0:
                self._print_string(f"{self._get_indent()}" + ")")
            else:
                self._print_string(")")

    def _print_region(self, region: Region):
        self._print_string("{\n")
        self._print_block(region.blocks[0], False)
        self.print_list(region.blocks[1:], self._print_block)
        if self._indent > 0:
            self._print_string(f"{self._get_indent()}" + "}\n")
        else:
            self._print_string("}")

    def _print_block(self, block: Block, print_args=True):
        if print_args:
            self._print_string(block.name)
            if len(block.arguments):
                self._print_string(" (")
                self.print_list(block.arguments, self.print)
                self._print_string(")")
            self._print_string(":\n")
        self._indent += 1
        self.print_list(block.operations, self.print, sep='')
        self._indent -= 1

    def _print_attributes(self, op: Operation):
        if len(op.attributes) > 0:
            self._print_string(" {")
            self.print_dict(op.attributes, self._print_attribute)
            self._print_string("}")

    def _print_op_type(self, op: Operation):
        input_types = [o.type for o in op.operands]
        output_types = [o.type for o in op.results]

        self._print_string("(")
        self.print_list(input_types, self._print_type)
        self._print_string(")")

        self._print_string(" -> ")
        if len(output_types) == 1:
            self._print_type(output_types[0])
        else:
            self._print_string("(")
            self.print_list(output_types, self._print_type)
            self._print_string(")")

    def _print_attribute(self, name: str, attr: Attribute):
        self._print_string(f"{name} = ")
        if isinstance(attr, IntegerAttr | FloatAttr):
            self._print_string(f"{attr.value}: {self._get_type_str(attr.type)}")
        elif isinstance(attr, BoolAttr):
            self._print_string("true" if attr.value else "false")
        elif isinstance(attr, NoneAttr):
            self._print_string("none")
        elif isinstance(attr, StringAttr):
            self._print_string(attr.value)
        elif isinstance(attr, DenseIntOrFPElementsAttr):
            self._print_string(f"dense<{attr.value}>: {self._get_type_str(attr.type)}")
        elif isinstance(attr, ArrayAttr):
            self._print_string(f"array<{self._get_type_str(attr.type.element_type)}: {', '.join([str(o) for o in attr.value])}>")

    def _print_type(self, t: IRType):
        supported_type = [IntegerType, FloatType, BoolType, NoneType,
                          RankedTensorType, UnrankedTensorType]
        if type(t) in supported_type:
            self._print_string(self._get_type_str(t))
        else:
            self._print_string(f"unkown")
    
    def _get_type_str(self, t: IRType):
        if isinstance(t, IntegerType):
            return f"{'u' if not t.signedness else ''}i{t.bitwidth}"
        elif isinstance(t, FloatType):
            return f"f{t.bitwidth}"
        elif isinstance(t, BoolType):
            return f"i1"
        elif isinstance(t, NoneType):
            return "()"
        elif isinstance(t, RankedTensorType):
            shape_str = [str(o) for o in t.dims]
            elem_type_str = [self._get_type_str(t.element_type)]
            return f"tensor<{'x'.join(shape_str + elem_type_str)}>"
        elif isinstance(t, UnrankedTensorType):
            shape_str = ['*']
            elem_type_str = [self._get_type_str(t.element_type)]
            return f"tensor<{'x'.join(shape_str + elem_type_str)}>"
        else:
            return "undefined"

    def _print_func(self, op: FuncOp):
        sym_name = op.attributes['sym_name'].value
        prefix = self._get_indent()
        func_type = op.attributes['function_type']
        input_types = [self._get_type_str(o) for o in func_type.input_types]
        output_types = [self._get_type_str(o) for o in func_type.output_types]
        arg_str = ', '.join([f'%arg{i}: {o}' for i, o in enumerate(input_types)])
        output_type_str = ', '.join(output_types)

        self._print_string(f"{prefix}{op.name} @{sym_name}({arg_str}) -> ({output_type_str}) ")
        self._print_region(op.regions[0])

    def _print_ret(self, op: ReturnOp):
        prefix = self._get_indent()
        if len(op.operands) > 0:
            rets = ','.join([o.name for o in op.operands])
            ret_types = ','.join([self._get_type_str(o.type) for o in op.operands])
            ret = f' {rets}: {ret_types}'
        else:
            ret = ''
        self._print_string(f"{prefix}return{ret}\n")

    def _print_call(self, op: CallOp):
        prefix = self._get_indent()
        results = ','.join([o.name for o in op.results])
        if results != '':
            results += ' = '
        sym_name = op.attributes['callee'].sym_name.value
        operands = ','.join([o.name for o in op.operands])
        operand_types = ','.join([self._get_type_str(o.type) for o in op.operands])
        output_types = ','.join([self._get_type_str(o.type) for o in op.results])
        self._print_string(f"{prefix}{results}func.call @{sym_name}({operands}) : ({operand_types}) -> ({output_types})\n")
