
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
import numpy as np

from .bytecode import ByteCodeModule, OpCode, TypeCode
from .dispatch import DispatchHintParser, DispatchInfo, get_dispatch_function


class StackFrame:
    def __init__(self, variable_size) -> None:
        self.operand_stack = []
        self.local_variable = [None] * variable_size


class VM:
    def __init__(self, bc: ByteCodeModule) -> None:
        self.bc = bc
        self.stackframe = StackFrame(self.bc.variable_size)
        self.current_stackframe = self.stackframe # should only main stackframe
    
    def push_operand(self, val: Any):
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
                tcode = self.bc.code[idx+1]
                if tcode == TypeCode.scalar:
                    val = self.bc.consts[self.bc.code[idx+2]]
                    self.push_operand(val)
                    idx += 3
                elif tcode == TypeCode.tensor:
                    num_ranks = self.bc.code[idx+2]
                    shape = [self.bc.code[i] for i in range(idx+3, idx+3+num_ranks)]
                    # element type: idx+3+num_ranks, unused here
                    val = self.bc.consts[self.bc.code[idx+3+num_ranks+1]]
                    if isinstance(val, Iterable):
                        val = np.reshape(val, shape)
                    else:
                        val = np.full(shape, val)
                    self.push_operand(val)
                    idx += 3+num_ranks+2
                else:
                    raise ValueError(f"Unsupported data type: {tcode}")
            elif self.bc.code[idx] == OpCode.invokestatic:
                hint = self.bc.consts[self.bc.code[idx+1]]
                disp_info = DispatchHintParser(hint).parse()
                disp_info.target = self.bc.target
                self.invoke(disp_info)
                idx += 2
            elif self.bc.code[idx] == OpCode.iload:
                self.push_operand(self.current_stackframe.local_variable[self.bc.code[idx+1]])
                idx += 2
            elif self.bc.code[idx] == OpCode.iload_0:
                self.push_operand(self.current_stackframe.local_variable[0])
                idx += 1
            elif self.bc.code[idx] == OpCode.iload_1:
                self.push_operand(self.current_stackframe.local_variable[1])
                idx += 1
            elif self.bc.code[idx] == OpCode.iload_2:
                self.push_operand(self.current_stackframe.local_variable[2])
                idx += 1
            elif self.bc.code[idx] == OpCode.iload_3:
                self.push_operand(self.current_stackframe.local_variable[3])
                idx += 1
            elif self.bc.code[idx] == OpCode.fload:
                self.push_operand(self.current_stackframe.local_variable[self.bc.code[idx+1]])
                idx += 2
            elif self.bc.code[idx] == OpCode.fload_0:
                self.push_operand(self.current_stackframe.local_variable[0])
                idx += 1
            elif self.bc.code[idx] == OpCode.fload_1:
                self.push_operand(self.current_stackframe.local_variable[1])
                idx += 1
            elif self.bc.code[idx] == OpCode.fload_2:
                self.push_operand(self.current_stackframe.local_variable[2])
                idx += 1
            elif self.bc.code[idx] == OpCode.fload_3:
                self.push_operand(self.current_stackframe.local_variable[3])
                idx += 1
            elif self.bc.code[idx] == OpCode.aload:
                self.push_operand(self.current_stackframe.local_variable[self.bc.code[idx+1]])
                idx += 2
            elif self.bc.code[idx] == OpCode.aload_0:
                self.push_operand(self.current_stackframe.local_variable[0])
                idx += 1
            elif self.bc.code[idx] == OpCode.aload_1:
                self.push_operand(self.current_stackframe.local_variable[1])
                idx += 1
            elif self.bc.code[idx] == OpCode.aload_2:
                self.push_operand(self.current_stackframe.local_variable[2])
                idx += 1
            elif self.bc.code[idx] == OpCode.aload_3:
                self.push_operand(self.current_stackframe.local_variable[3])
                idx += 1
            elif self.bc.code[idx] == OpCode.astore:
                self.current_stackframe.local_variable[self.bc.code[idx+1]] = self.pop_operand()
                idx += 2
            elif self.bc.code[idx] == OpCode.astore_0:
                self.current_stackframe.local_variable[0] = self.pop_operand()
                idx += 1
            elif self.bc.code[idx] == OpCode.astore_1:
                self.current_stackframe.local_variable[1] = self.pop_operand()
                idx += 1
            elif self.bc.code[idx] == OpCode.astore_2:
                self.current_stackframe.local_variable[2] = self.pop_operand()
                idx += 1
            elif self.bc.code[idx] == OpCode.astore_3:
                self.current_stackframe.local_variable[3] = self.pop_operand()
                idx += 1
            elif self.bc.code[idx] == OpCode.istore:
                self.current_stackframe.local_variable[self.bc.code[idx+1]] = self.pop_operand()
                idx += 2
            elif self.bc.code[idx] == OpCode.istore_0:
                self.current_stackframe.local_variable[0] = self.pop_operand()
                idx += 1
            elif self.bc.code[idx] == OpCode.istore_1:
                self.current_stackframe.local_variable[1] = self.pop_operand()
                idx += 1
            elif self.bc.code[idx] == OpCode.istore_2:
                self.current_stackframe.local_variable[2] = self.pop_operand()
                idx += 1
            elif self.bc.code[idx] == OpCode.istore_3:
                self.current_stackframe.local_variable[3] = self.pop_operand()
                idx += 1
            elif self.bc.code[idx] == OpCode.fstore:
                self.current_stackframe.local_variable[self.bc.code[idx+1]] = self.pop_operand()
                idx += 2
            elif self.bc.code[idx] == OpCode.fstore_0:
                self.current_stackframe.local_variable[0] = self.pop_operand()
                idx += 1
            elif self.bc.code[idx] == OpCode.fstore_1:
                self.current_stackframe.local_variable[1] = self.pop_operand()
                idx += 1
            elif self.bc.code[idx] == OpCode.fstore_2:
                self.current_stackframe.local_variable[2] = self.pop_operand()
                idx += 1
            elif self.bc.code[idx] == OpCode.fstore_3:
                self.current_stackframe.local_variable[3] = self.pop_operand()
                idx += 1
            else:
                raise NotImplementedError(f"opcode: {self.bc.code[idx]}, index: {idx}")
    
    def invoke(self, info: DispatchInfo):
        num_args = len(info.sig[0])
        args = [self.pop_operand() for _ in range(num_args)]
        # Due to the first-in-last-out stack,
        # a reversal is required to get the correct order
        args.reverse()
        func = get_dispatch_function(info)
        out = func(*args)
        if isinstance(out, list | tuple):
            for o in out:
                self.push_operand(o)
        else:
            self.push_operand(out)
