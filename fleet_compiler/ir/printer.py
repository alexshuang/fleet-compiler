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
                op.parent(op)
            else:
                self._print_results(op)
                self._print_op_with_default_format(op)
        elif isinstance(op, Region):
            self._print_region(op)
        elif isinstance(op, Block):
            self._print_block(op)
        elif isinstance(op, Value):
            self._print_string(op.name)

    def _print_results(self, op: Operation):
        self._print_string("  " * self._indent)
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
        self._print_string(" (")
        self.print_list(op.regions, self._print_region)
        self._print_string(")")

    def _print_region(self, region: Region):
        self._print_string("{\n")
        self._print_block(region.blocks[0], False)
        self.print_list(region.blocks[1:], self._print_block)
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
        def _get_type(t: IRType):
            if isinstance(t, IntegerType):
                return f"{'u' if not t.signedness else ''}i{t.bitwidth}"
            elif isinstance(t, FloatType):
                return f"f{t.bitwidth}"
            else:
                return ""
            
        self._print_string(f"{name} = ")
        if isinstance(attr, IntegerAttr):
            self._print_string(f"{attr.value}: {_get_type(attr.type)}")
        elif isinstance(attr, FloatAttr):
            self._print_string(f"{attr.value}: {_get_type(attr.type)}")
        elif isinstance(attr, BoolAttr):
            self._print_string("true" if attr.value else "false")
        elif isinstance(attr, NoneAttr):
            self._print_string("none")

    def _print_type(self, t: IRType):
        if isinstance(t, IntegerType):
            self._print_string(f"{'u' if not t.signedness else ''}i{t.bitwidth}")
        elif isinstance(t, FloatType):
            self._print_string(f"f{t.bitwidth}")
        elif isinstance(t, BoolType):
            self._print_string(f"i1")
        elif isinstance(t, NoneType):
            self._print_string(f"()")
        else:
            self._print_string(f"unkown")
