
# ===- bytecode.py -------------------------------------------------------------
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
from dataclasses import dataclass, field
from enum import Enum, auto

from fleet_compiler.ir.core import *
from fleet_compiler.ir.dialects.builtin import *
from fleet_compiler.ir.dialects.func import FlatSymbolRefAttr


def get_shape(type: IRType):
    if isinstance(type, RankedTensorType):
        bw = type.element_type.bitwidth
        dtype = f'i{bw}' if isinstance(type.element_type, IntegerType) else f'f{bw}'
        shape = 'x'.join([str(p) for p in type.dims])
        return f'{shape}x{dtype}'
    else:
        bw = type.bitwidth
        dtype = f'i{bw}' if isinstance(type, IntegerType) else f'f{bw}'
        return dtype


class OpCode(Enum):
    iconst_m1 = 1
    iconst_0 = 2
    iconst_1 = 3
    iconst_2 = 4
    iconst_3 = 5
    iconst_4 = 6
    iconst_5 = 7
    fconst_0 = 8
    fconst_1 = 9
    fconst_2 = 10
    ldc = 11
    invokestatic = 12


@dataclass
class ByteCodeModule:
    consts: list = field(default_factory=list)
    code: list = field(default_factory=list)
    target: str = field(default="python")


class ByteCodeConverter:
    def __init__(self, module: ModuleOp) -> None:
        self.m = module
        self.bc = ByteCodeModule()
        self.const_cache: dict[Any, int] = {}

    def cache_check(self, val: Any):
        return isinstance(val, int | float | str)
        
    def add_code(self, code: Any):
        self.bc.code.append(code)

    def add_const(self, val: Any):
        # return the position of value in const
        if (idx := self.index_of(val)) < 0:
            self.bc.consts.append(val)
            idx = len(self.bc.consts) - 1
            if self.cache_check(val):
                self.const_cache[val] = idx
            return idx
        else:
            return idx
    
    def index_of(self, val: Any):
        return self.const_cache[val] if self.cache_check(val) and val in self.const_cache else -1

    def _handle_operand(self, operand: OpResult):
        op = operand.owner()
        if op.name == "vm.rodata":
            self.add_code(OpCode.ldc)
            self.add_code(self.add_const(op.attributes['value'].value))
        elif op.name == "vm.const.i32":
            val = op.attributes['value'].value
            if val == -1:
                self.add_code(OpCode.iconst_m1)
            elif val == 0:
                self.add_code(OpCode.iconst_0)
            elif val == 1:
                self.add_code(OpCode.iconst_1)
            elif val == 2:
                self.add_code(OpCode.iconst_2)
            elif val == 3:
                self.add_code(OpCode.iconst_3)
            elif val == 4:
                self.add_code(OpCode.iconst_4)
            elif val == 5:
                self.add_code(OpCode.iconst_5)
            else:
                self.add_code(OpCode.ldc)
                self.add_code(self.add_const(val))
        elif op.name == "vm.const.f32":
            val = op.attributes['value'].value
            if val == 0.0:
                self.add_code(OpCode.fconst_0)
            elif val == 1.0:
                self.add_code(OpCode.fconst_1)
            elif val == 2.0:
                self.add_code(OpCode.fconst_2)
            else:
                self.add_code(OpCode.ldc)
                self.add_code(self.add_const(val))
        elif op.name == "vm.call":
            pass
        else:
            raise NotImplementedError
    
    def _get_vm_invoke_signature(self, op: Operation):
        def _type_to_abi(type: IRType):
            if isinstance(type, FloatType):
                assert type.bitwidth in [16, 32, 64]
                if type.bitwidth == 64:
                    return 'd'
                elif type.bitwidth == 32:
                    return 'f'
                elif type.bitwidth == 16:
                    return 'h'
            elif isinstance(type, IntegerType):
                assert type.bitwidth in [1, 32, 64]
                if type.bitwidth == 64:
                    return 'I'
                elif type.bitwidth == 32:
                    return 'i'
                elif type.bitwidth == 1:
                    return 'b'
            elif isinstance(type, BoolType):
                return 'b'
            elif isinstance(type, RankedTensorType | UnrankedTensorType):
                return 'r'
            else:
                raise NotImplementedError
        res = ''        
        if op.operands:
            res += ''.join([_type_to_abi(o.type) for o in op.operands])
        if op.results:
            res += '_' + ''.join([_type_to_abi(o.type) for o in op.results])
        return res

    def convert(self):
        assert len(self.m.regions) == 1
        assert len(self.m.regions[0].blocks) == 1
        self.bc.target = self.m.attributes['target_info'].value['target_backend'].value
        for op in self.m.operations:
            if op.name == "vm.call":
                # create callee function name: e.g. matmul_sig_xxx_ins_xxx_outs_xxx_...
                attrs = op.attributes.copy()
                callee = attrs.pop('callee').sym_name.value
                ins = [get_shape(o.type) for o in op.operands]
                outs = [get_shape(o.type) for o in op.results]
                sig = self._get_vm_invoke_signature(op)
                callee += "_sig_" + sig
                callee += "_ins_" + '_'.join(ins)
                if outs:
                    callee += "_outs_" + '_'.join(outs)
                callee = callee.replace('.', '_')
                    
                for k, v in attrs.items():
                    if isinstance(v, FlatSymbolRefAttr):
                        v = v.sym_name.value
                    else:
                        v = v.value
                    if isinstance(v, Iterable):
                        v = '_'.join([str(o) for o in v])
                    elif not isinstance(v, str):
                        v = str(v)
                    callee += f"_{k}_{v}"

                for o in op.operands:
                    self._handle_operand(o)
                self.add_code(OpCode.invokestatic)
                self.add_code(self.add_const(callee))

        return self.bc
