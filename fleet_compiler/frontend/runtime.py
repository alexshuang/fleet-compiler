# ===- runtime.py -------------------------------------------------------------
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

from .lexer import Op
from .ast import *
from .ops import Operation
from .symbolic import OperatorSymbol
import contextlib
from typing import Any


class RetVal:
    def __init__(self, value=None) -> None:
        self.value = value


class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class CallStack(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self._stack = []

    def push(self, frame: StackFrame):
        self._stack.append(frame)

    def pop(self):
        return self._stack.pop()

    def get(self, index: int = None):
        return self._stack[index] if index else self._stack[-1]

    def size(self):
        return len(self._stack)


class StackFrame(contextlib.AbstractContextManager):
    def __init__(self) -> None:
        self.variables = {} # local variables
        self.ret_val = None # return value

    def update(self, name: str, value):
        self.variables[name] = value

    def get(self, name: str):
        return self.variables[name] if name in self.variables else None

    def __enter__(self) -> Any:
        CallStack().push(self)
        return CallStack().get()

    def __exit__(self, __exc_type: type[BaseException] | None,
                 __exc_value: BaseException | None,
                 __traceback: contextlib.TracebackType | None) -> bool | None:
        return CallStack().pop()


class Interpreter(AstVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.call_stack = CallStack()
        self.ops = Operation()
    
    def visitModule(self, node: AstModule):
        with StackFrame():
            return super().visitModule(node)

    def visitBlock(self, node: Block):
        def should_run(node):
            # Instructions that should be executed
            return not isinstance(node, FunctionDef)

        ret = None
        for o in node.stmts:
            if should_run(o):
                ret = self.visit(o)
                if isinstance(ret, RetVal):
                    if ret.value is not None: # not return by BlockEnd
                        self.set_return_value(ret.value)
                    return ret
    
    def visitBlockEnd(self, node: BlockEnd):
        return RetVal()
    
    def visitFunctionCall(self, node: FunctionCall):
        if node.sym:
            if isinstance(node.sym, OperatorSymbol):
                # third-party or built-in operators
                arg_list = self.visitArgumentList(node.arg_list)
                args, kwargs = [], {}
                for o in arg_list:
                    if isinstance(o, dict):
                        kwargs.update(o)
                    else:
                        args.append(o)
                op_name = node.sym.op_name
                if self.ops.has(op_name):
                    return self.ops.lookup(op_name)(args, kwargs)
                else:
                    raise TypeError(f"Interpreter: Unsupport operation {node.name}")
            else: # function define
                with StackFrame():
                    # args
                    func = node.sym.node
                    params = func.signature.param_list.params
                    args = node.arg_list.args
                    num_args = len(args)
                    num_params = len(params)
                    if num_params < num_args:
                        raise TypeError(f"Interpreter: Expect {num_params} arguments but got {num_args}")

                    names, values = [], []
                    for i, arg in enumerate(args):
                        names.append(params[i].name if isinstance(arg, PositionalArgument) else arg.name)
                        values.append(self.visit(arg.value))
                    # args with default-value
                    for p in params[num_args:]:
                        names.append(p.name)
                        values.append(self.visit(p.init))

                    # set variable values
                    for k, v in zip(names, values):
                        self.update_variable_value(k, v)

                    # body
                    ret = self.visitBlock(func.block)
                return self.call_stack.get().ret_val
        else:
            raise TypeError(f"Interpreter: Unsupport operation {node.name}")
    
    def visitArgumentList(self, node: ArgumentList):
        args = [self.visit(o) for o in node.args if o]
        return args if len(args) > 0 else ""
    
    def visitVariableDef(self, node: VariableDef):
        self.update_variable_value(node.name, self.visit(node.init))
    
    def visitVariable(self, node: Variable):
        return self.get_variable_value(node.name)

    def visitReturnStatement(self, node: ReturnStatement):
        value = self.visit(node.ret) if node.ret else None
        self.set_return_value(value)
        return RetVal(value)

    def visitParamentDef(self, node: ParamentDef):
        self.update_variable_value(node.name, self.visit(node.init))
        return super().visitParamentDef(node)
    
    def visitBinary(self, node: Binary):
        if node.op == Op.Assign:
            return self.visit(node.exp2)
        elif node.op == Op.Plus:
            return self.visit(node.exp1) + self.visit(node.exp2)
        elif node.op == Op.Minus:
            return self.visit(node.exp1) - self.visit(node.exp2)
        elif node.op == Op.Multiply:
            return self.visit(node.exp1) * self.visit(node.exp2)
        elif node.op == Op.Divide:
            return self.visit(node.exp1) / self.visit(node.exp2)
        elif node.op == Op.AT:
            return self.visit(node.exp1) @ self.visit(node.exp2)
        elif node.op == Op.Power:
            return self.visit(node.exp1) ** self.visit(node.exp2)
        elif node.op == Op.EQ:
            return self.visit(node.exp1) == self.visit(node.exp2)
        elif node.op == Op.NE:
            return self.visit(node.exp1) != self.visit(node.exp2)
        elif node.op == Op.GT:
            return self.visit(node.exp1) > self.visit(node.exp2)
        elif node.op == Op.GE:
            return self.visit(node.exp1) >= self.visit(node.exp2)
        elif node.op == Op.LT:
            return self.visit(node.exp1) < self.visit(node.exp2)
        elif node.op == Op.LE:
            return self.visit(node.exp1) <= self.visit(node.exp2)
        else:
            raise TypeError(f"Interpreter: Unsupport operator {node.op}")
    
    def visitIfStatement(self, node: IfStatement):
        for b in node.branches:
            # else branch
            if b.cond is None:
                with StackFrame():
                    ret = self.visitBlock(b.block)
                return ret
            elif self.visit(b.cond): # then branch
                with StackFrame():
                    ret = self.visitBlock(b.block)
                return ret

    def visitSliceStatement(self, node: SliceStatement):
        slice_str = self.visitSlice(node.slice)
        op_name = node.sym.op_name
        if self.ops.has(op_name):
            return self.ops.lookup(op_name)([self.get_variable_value(node.name), slice_str], {})

    def visitSlice(self, node: Slice):
        slices = [str(self.visit(o)) for o in node.exps]
        if node.omitted_first_dim:
            slices = [''] + slices
        if node.omitted_last_dim:
            slices.append('')
        return ':'.join(slices)

    def visitBranch(self, node: Branch):
        return super().visitBranch(node)

    def visitListStatement(self, node: ListStatement):
        return self.visitListContent(node.content)

    def visitListContent(self, node: ListContent):
        return [self.visit(o) for o in node.exps]

    def get_variable_value(self, name: str):
        value = None
        for i in range(self.call_stack.size() - 1, -1, -1):
            value = self.call_stack.get(i).get(name)
            if value is not None:
                break
        if value is None:
            raise NameError(f"name {name} is not defined")
        return value

    def update_variable_value(self, name: str, value):
        if frame := self.call_stack.get():
            frame.update(name, value)

    def set_return_value(self, value):
        self.call_stack.get(-2).ret_val = value

