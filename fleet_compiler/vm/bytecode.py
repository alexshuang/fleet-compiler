
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
    iload = 13
    iload_0 = 14
    iload_1 = 15
    iload_2 = 16
    iload_3 = 17
    istore = 18
    istore_0 = 19
    istore_1 = 20
    istore_2 = 21
    istore_3 = 22
    fload = 23
    fload_0 = 24
    fload_1 = 25
    fload_2 = 26
    fload_3 = 27
    fstore = 28
    fstore_0 = 29
    fstore_1 = 30
    fstore_2 = 31
    fstore_3 = 32
    aload = 33
    aload_0 = 34
    aload_1 = 35
    aload_2 = 36
    aload_3 = 37
    astore = 38
    astore_0 = 39
    astore_1 = 40
    astore_2 = 41
    astore_3 = 42


class TypeCode(Enum):
    scalar = 1
    tensor = 2
    string = 3
    int32 = 4
    float32 = 5
    int64 = 6
    float64 = 7


def element_type_to_code(elem_type: IRType):
    if isinstance(elem_type, FloatType):
        if elem_type.bitwidth == 32:
            return TypeCode.float32
        elif elem_type.bitwidth == 64:
            return TypeCode.float64
    elif isinstance(elem_type, IntegerType):
        if elem_type.bitwidth == 32:
            return TypeCode.int32
        elif elem_type.bitwidth == 64:
            return TypeCode.int64
    raise NotImplementedError


@dataclass
class ByteCodeModule:
    consts: list = field(default_factory=list)
    code: list = field(default_factory=list)
    target: str = field(default="python")
    variable_size: int = field(default=0)


class ByteCodeConverter:
    curr_var_index = 0
    
    def __init__(self, module: ModuleOp) -> None:
        self.m = module
        self.bc = ByteCodeModule()
        self.const_cache: dict[Any, int] = {}
        self.variable_cache: dict[str, int] = {}

    def cache_check(self, val: Any):
        return isinstance(val, int | float | str)
        
    def add_code(self, code: Any):
        self.bc.code.append(code)

    def add_const(self, val: Any):
        # return the position of value in const
        if (idx := self.const_index_of(val)) < 0:
            self.bc.consts.append(val)
            idx = len(self.bc.consts) - 1
            if self.cache_check(val):
                self.const_cache[val] = idx
            return idx
        else:
            return idx

    def add_variable(self, name: str):
        self.variable_cache[name] = self.curr_var_index
        self.curr_var_index += 1

    def const_index_of(self, val: Any):
        return self.const_cache.get(val, -1) if self.cache_check(val) else -1

    def var_index_of(self, name: str):
        return self.variable_cache.get(name, -1)

    def _handle_operand(self, operand: OpResult):
        op = operand.owner()
        if op.name == "vm.rodata":
            self.add_code(OpCode.ldc)
            self.add_code(TypeCode.tensor) # datatype
            dims = op.results[0].type.dims
            self.add_code(len(dims)) # num_dims
            num_elements = 0
            for dim in dims:
                self.add_code(dim)
                num_elements = dim if num_elements == 0 else num_elements * dim
            # datatype
            self.add_code(element_type_to_code(op.results[0].type.element_type))
            value = op.attributes['value'].value
            if isinstance(value, Iterable):
                assert np.array(value).size == num_elements, \
                    f"Unmatched value shape with expected: {np.array(value).shape} vs. {dims}"
            self.add_code(self.add_const(value))
            self.add_variable(op.results[0].name)
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
                self.add_code(TypeCode.scalar)
                self.add_code(self.add_const(val))
            self.add_variable(op.results[0].name)
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
                self.add_code(TypeCode.scalar)
                self.add_code(self.add_const(val))
            self.add_variable(op.results[0].name)
        elif op.name == "vm.call":
            def add_load_code(type: IRType, idx: int):
                if isinstance(type, IntegerType):
                    if idx == 0:
                        self.add_code(OpCode.iload_0)
                    elif idx == 1:
                        self.add_code(OpCode.iload_1)
                    elif idx == 2:
                        self.add_code(OpCode.iload_2)
                    elif idx == 3:
                        self.add_code(OpCode.iload_3)
                    else:
                        self.add_code(OpCode.iload)
                        self.add_code(idx)
                elif isinstance(type, FloatType):
                    if idx == 0:
                        self.add_code(OpCode.fload_0)
                    elif idx == 1:
                        self.add_code(OpCode.fload_1)
                    elif idx == 2:
                        self.add_code(OpCode.fload_2)
                    elif idx == 3:
                        self.add_code(OpCode.fload_3)
                    else:
                        self.add_code(OpCode.fload)
                        self.add_code(idx)
                elif isinstance(type, RankedTensorType):
                    if idx == 0:
                        self.add_code(OpCode.aload_0)
                    elif idx == 1:
                        self.add_code(OpCode.aload_1)
                    elif idx == 2:
                        self.add_code(OpCode.aload_2)
                    elif idx == 3:
                        self.add_code(OpCode.aload_3)
                    else:
                        self.add_code(OpCode.aload)
                        self.add_code(idx)

            assert len(op.results) == 1
            assert (idx := self.var_index_of(op.results[0].name)) >= 0
            add_load_code(op.results[0].type, idx)
        else:
            raise NotImplementedError
    
    def _handle_result(self, result: OpResult):
        # There is no overwriting of data due to SSA value,
        # no memory optimization here
        idx = self.curr_var_index
        if isinstance(result.type, IntegerType):
            if idx == 0:
                self.add_code(OpCode.istore_0)
            elif idx == 1:
                self.add_code(OpCode.istore_1)
            elif idx == 2:
                self.add_code(OpCode.istore_2)
            elif idx == 3:
                self.add_code(OpCode.istore_3)
            else:
                self.add_code(OpCode.istore)
                self.add_code(idx)
        elif isinstance(result.type, FloatType):
            if idx == 0:
                self.add_code(OpCode.fstore_0)
            elif idx == 1:
                self.add_code(OpCode.fstore_1)
            elif idx == 2:
                self.add_code(OpCode.fstore_2)
            elif idx == 3:
                self.add_code(OpCode.fstore_3)
            else:
                self.add_code(OpCode.fstore)
                self.add_code(idx)
        elif isinstance(result.type, RankedTensorType):
            if idx == 0:
                self.add_code(OpCode.astore_0)
            elif idx == 1:
                self.add_code(OpCode.astore_1)
            elif idx == 2:
                self.add_code(OpCode.astore_2)
            elif idx == 3:
                self.add_code(OpCode.astore_3)
            else:
                self.add_code(OpCode.astore)
                self.add_code(idx)
        else:
            raise NotImplementedError
        
        self.add_variable(result.name)

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
        if 'target_info' in self.m.attributes:
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
                for o in op.results:
                    self._handle_result(o)

        self.bc.variable_size = self.curr_var_index

        return self.bc
