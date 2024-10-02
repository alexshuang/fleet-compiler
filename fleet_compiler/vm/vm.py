
# ===- vm.py -------------------------------------------------------------
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
from typing import Any, Iterable
from dataclasses import dataclass, field

from .bytecode import ByteCodeModule, OpCode
from .dispatch import DispatchHintParser, DispatchInfo, get_dispatch_function


@dataclass
class StackFrame:
    operand_stack: list = field(default_factory=list)


class VM:
    def __init__(self, bc: ByteCodeModule) -> None:
        self.bc = bc
        self.stackframe = StackFrame()
        self.current_stackframe = self.stackframe # should only main stackframe
    
    def push_operand(self, val: Any):
        if isinstance(val, Iterable):
            self.current_stackframe.operand_stack.extend(val)
        else:
            self.current_stackframe.operand_stack.append(val)

    def pop_operand(self):
        return self.current_stackframe.operand_stack.pop()

    def run(self):
        idx = 0
        code_size = len(self.bc.code)
        while idx < code_size:
            if self.bc.code[idx] == OpCode.fconst_0:
                self.push_operand(0.)
                idx += 1
            elif self.bc.code[idx] == OpCode.fconst_1:
                self.push_operand(1.)
                idx += 1
            elif self.bc.code[idx] == OpCode.fconst_2:
                self.push_operand(2.)
                idx += 1
            elif self.bc.code[idx] == OpCode.iconst_m1:
                self.push_operand(-1)
                idx += 1
            elif self.bc.code[idx] == OpCode.iconst_0:
                self.push_operand(0)
                idx += 1
            elif self.bc.code[idx] == OpCode.iconst_1:
                self.push_operand(1)
                idx += 1
            elif self.bc.code[idx] == OpCode.iconst_2:
                self.push_operand(2)
                idx += 1
            elif self.bc.code[idx] == OpCode.iconst_3:
                self.push_operand(3)
                idx += 1
            elif self.bc.code[idx] == OpCode.iconst_4:
                self.push_operand(4)
                idx += 1
            elif self.bc.code[idx] == OpCode.iconst_5:
                self.push_operand(5)
                idx += 1
            elif self.bc.code[idx] == OpCode.ldc:
                val = self.bc.consts[self.bc.code[idx+1]]
                self.push_operand(val)
                idx += 2
            elif self.bc.code[idx] == OpCode.invokestatic:
                hint = self.bc.consts[self.bc.code[idx+1]]
                disp_info = DispatchHintParser(hint).parse()
                disp_info.target = self.bc.target
                self.invoke(disp_info)
                idx += 2
            else:
                raise NotImplementedError(f"opcode: {self.bc.code[idx]}, index: {idx}")
    
    def invoke(self, info: DispatchInfo):
        num_args = len(info.sig[0])
        args = [self.pop_operand() for _ in range(num_args)]
        func = get_dispatch_function(info)
        out = func(*args)
        self.push_operand(out)
